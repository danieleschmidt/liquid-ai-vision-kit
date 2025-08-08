#!/usr/bin/env python3
"""
Quantum System Audit Logging and Security Monitoring
Comprehensive audit trail with tamper detection and compliance features
"""

import time
import json
import logging
import hashlib
import hmac
import threading
import queue
import os
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
import gzip
import shutil
from concurrent.futures import ThreadPoolExecutor
import uuid
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import base64


class AuditEventType(Enum):
    """Types of audit events"""
    SYSTEM_START = "system_start"
    SYSTEM_STOP = "system_stop"
    TASK_SUBMITTED = "task_submitted"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    RESOURCE_ALLOCATED = "resource_allocated"
    RESOURCE_DEALLOCATED = "resource_deallocated"
    SECURITY_VIOLATION = "security_violation"
    ACCESS_GRANTED = "access_granted"
    ACCESS_DENIED = "access_denied"
    CONFIGURATION_CHANGED = "configuration_changed"
    DATA_ACCESS = "data_access"
    ERROR_OCCURRED = "error_occurred"
    PERFORMANCE_ALERT = "performance_alert"
    CACHE_OPERATION = "cache_operation"
    VALIDATION_FAILED = "validation_failed"
    AUTHENTICATION = "authentication"
    AUTHORIZATION = "authorization"
    QUANTUM_STATE_CHANGE = "quantum_state_change"


class AuditLevel(Enum):
    """Audit logging levels"""
    DEBUG = 0
    INFO = 1
    WARNING = 2
    ERROR = 3
    CRITICAL = 4
    SECURITY = 5


@dataclass
class AuditEvent:
    """Individual audit event"""
    event_id: str
    event_type: AuditEventType
    timestamp: float
    level: AuditLevel
    source: str
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    ip_address: Optional[str] = None
    user_agent: Optional[str] = None
    request_id: Optional[str] = None
    
    # Security fields
    integrity_hash: Optional[str] = None
    signature: Optional[str] = None
    
    def __post_init__(self):
        """Generate integrity hash after initialization"""
        if not self.event_id:
            self.event_id = str(uuid.uuid4())
        
        # Calculate integrity hash
        self.integrity_hash = self._calculate_integrity_hash()
    
    def _calculate_integrity_hash(self) -> str:
        """Calculate integrity hash for tamper detection"""
        # Create hash of core event data (excluding integrity_hash and signature)
        event_data = {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'level': self.level.value,
            'source': self.source,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'message': self.message,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id': self.request_id
        }
        
        event_json = json.dumps(event_data, sort_keys=True, separators=(',', ':'))
        return hashlib.sha256(event_json.encode()).hexdigest()
    
    def verify_integrity(self) -> bool:
        """Verify event integrity"""
        if not self.integrity_hash:
            return False
        
        current_hash = self._calculate_integrity_hash()
        return current_hash == self.integrity_hash
    
    def sign(self, secret_key: str):
        """Sign the event with HMAC"""
        if not self.integrity_hash:
            self.integrity_hash = self._calculate_integrity_hash()
        
        self.signature = hmac.new(
            secret_key.encode(),
            self.integrity_hash.encode(),
            hashlib.sha256
        ).hexdigest()
    
    def verify_signature(self, secret_key: str) -> bool:
        """Verify event signature"""
        if not self.signature or not self.integrity_hash:
            return False
        
        expected_signature = hmac.new(
            secret_key.encode(),
            self.integrity_hash.encode(),
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(self.signature, expected_signature)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'event_id': self.event_id,
            'event_type': self.event_type.value,
            'timestamp': self.timestamp,
            'level': self.level.value,
            'source': self.source,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'message': self.message,
            'details': self.details,
            'ip_address': self.ip_address,
            'user_agent': self.user_agent,
            'request_id': self.request_id,
            'integrity_hash': self.integrity_hash,
            'signature': self.signature
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AuditEvent':
        """Create from dictionary"""
        # Convert string enums back to enum objects
        event_type = AuditEventType(data['event_type'])
        level = AuditLevel(data['level'])
        
        return cls(
            event_id=data['event_id'],
            event_type=event_type,
            timestamp=data['timestamp'],
            level=level,
            source=data['source'],
            user_id=data.get('user_id'),
            session_id=data.get('session_id'),
            message=data.get('message', ''),
            details=data.get('details', {}),
            ip_address=data.get('ip_address'),
            user_agent=data.get('user_agent'),
            request_id=data.get('request_id'),
            integrity_hash=data.get('integrity_hash'),
            signature=data.get('signature')
        )


class AuditStorage:
    """Base class for audit storage backends"""
    
    def write_event(self, event: AuditEvent):
        """Write audit event to storage"""
        raise NotImplementedError
    
    def read_events(self, start_time: float, end_time: float, 
                   event_types: Optional[List[AuditEventType]] = None) -> List[AuditEvent]:
        """Read audit events from storage"""
        raise NotImplementedError
    
    def archive_events(self, before_time: float):
        """Archive old events"""
        raise NotImplementedError


class FileAuditStorage(AuditStorage):
    """File-based audit storage with rotation and compression"""
    
    def __init__(self, base_dir: str = "/var/log/quantum_audit", 
                 max_file_size: int = 100 * 1024 * 1024,  # 100MB
                 retention_days: int = 90,
                 compress_old_files: bool = True):
        self.base_dir = Path(base_dir)
        self.max_file_size = max_file_size
        self.retention_days = retention_days
        self.compress_old_files = compress_old_files
        
        # Create directory if it doesn't exist
        self.base_dir.mkdir(parents=True, exist_ok=True)
        
        # Current log file
        self.current_file = self.base_dir / "audit.log"
        self.lock = threading.Lock()
    
    def write_event(self, event: AuditEvent):
        """Write event to current log file"""
        with self.lock:
            # Check if rotation is needed
            if self.current_file.exists() and self.current_file.stat().st_size > self.max_file_size:
                self._rotate_file()
            
            # Write event
            with open(self.current_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(event.to_dict()) + '\n')
    
    def read_events(self, start_time: float, end_time: float, 
                   event_types: Optional[List[AuditEventType]] = None) -> List[AuditEvent]:
        """Read events from log files"""
        events = []
        
        # Get all log files in date range
        log_files = self._get_log_files_in_range(start_time, end_time)
        
        for log_file in log_files:
            events.extend(self._read_events_from_file(log_file, start_time, end_time, event_types))
        
        return sorted(events, key=lambda e: e.timestamp)
    
    def _rotate_file(self):
        """Rotate current log file"""
        if not self.current_file.exists():
            return
        
        # Generate rotated filename with timestamp
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        rotated_file = self.base_dir / f"audit_{timestamp}.log"
        
        # Move current file
        shutil.move(str(self.current_file), str(rotated_file))
        
        # Compress if enabled
        if self.compress_old_files:
            compressed_file = str(rotated_file) + '.gz'
            with open(rotated_file, 'rb') as f_in:
                with gzip.open(compressed_file, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            rotated_file.unlink()  # Remove uncompressed file
    
    def _get_log_files_in_range(self, start_time: float, end_time: float) -> List[Path]:
        """Get log files that might contain events in the time range"""
        log_files = []
        
        # Add current file
        if self.current_file.exists():
            log_files.append(self.current_file)
        
        # Add rotated files
        for file_path in self.base_dir.glob("audit_*.log*"):
            # For simplicity, include all rotated files
            # In production, you might want to parse timestamps from filenames
            log_files.append(file_path)
        
        return log_files
    
    def _read_events_from_file(self, file_path: Path, start_time: float, end_time: float,
                              event_types: Optional[List[AuditEventType]]) -> List[AuditEvent]:
        """Read events from a single log file"""
        events = []
        
        try:
            # Handle compressed files
            if file_path.suffix == '.gz':
                open_func = gzip.open
                mode = 'rt'
            else:
                open_func = open
                mode = 'r'
            
            with open_func(file_path, mode, encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        event_data = json.loads(line)
                        event = AuditEvent.from_dict(event_data)
                        
                        # Filter by time range
                        if start_time <= event.timestamp <= end_time:
                            # Filter by event types if specified
                            if event_types is None or event.event_type in event_types:
                                events.append(event)
                    
                    except (json.JSONDecodeError, ValueError, KeyError) as e:
                        # Log corrupted entry but continue
                        logging.error(f"Error parsing audit event: {e}")
        
        except Exception as e:
            logging.error(f"Error reading audit file {file_path}: {e}")
        
        return events
    
    def archive_events(self, before_time: float):
        """Archive/delete old log files"""
        cutoff_timestamp = before_time
        
        for file_path in self.base_dir.glob("audit_*.log*"):
            try:
                # Check file modification time
                if file_path.stat().st_mtime < cutoff_timestamp:
                    file_path.unlink()
                    logging.info(f"Archived old audit file: {file_path}")
            except Exception as e:
                logging.error(f"Error archiving file {file_path}: {e}")


class EncryptedAuditStorage(FileAuditStorage):
    """Encrypted file-based audit storage"""
    
    def __init__(self, base_dir: str = "/var/log/quantum_audit_encrypted",
                 encryption_key: Optional[str] = None, **kwargs):
        super().__init__(base_dir, **kwargs)
        
        # Initialize encryption
        if encryption_key:
            self.fernet = self._create_fernet(encryption_key)
        else:
            # Generate a new key (should be saved securely)
            key = Fernet.generate_key()
            self.fernet = Fernet(key)
            
            # Save key to file (in production, use secure key management)
            key_file = self.base_dir / "encryption.key"
            with open(key_file, 'wb') as f:
                f.write(key)
    
    def _create_fernet(self, password: str) -> Fernet:
        """Create Fernet cipher from password"""
        password_bytes = password.encode()
        salt = b'quantum_audit_salt'  # In production, use random salt
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100000,
        )
        key = base64.urlsafe_b64encode(kdf.derive(password_bytes))
        return Fernet(key)
    
    def write_event(self, event: AuditEvent):
        """Write encrypted event to file"""
        with self.lock:
            # Check rotation
            if self.current_file.exists() and self.current_file.stat().st_size > self.max_file_size:
                self._rotate_file()
            
            # Encrypt and write event
            event_json = json.dumps(event.to_dict())
            encrypted_data = self.fernet.encrypt(event_json.encode())
            
            with open(self.current_file, 'ab') as f:
                # Write length prefix + encrypted data + newline
                data_length = len(encrypted_data)
                f.write(data_length.to_bytes(4, 'big'))
                f.write(encrypted_data)
                f.write(b'\n')
    
    def _read_events_from_file(self, file_path: Path, start_time: float, end_time: float,
                              event_types: Optional[List[AuditEventType]]) -> List[AuditEvent]:
        """Read encrypted events from file"""
        events = []
        
        try:
            # Handle compressed files
            if file_path.suffix == '.gz':
                with gzip.open(file_path, 'rb') as f:
                    data = f.read()
            else:
                with open(file_path, 'rb') as f:
                    data = f.read()
            
            # Parse encrypted entries
            offset = 0
            while offset < len(data):
                if offset + 4 > len(data):
                    break
                
                # Read length
                length = int.from_bytes(data[offset:offset+4], 'big')
                offset += 4
                
                if offset + length + 1 > len(data):
                    break
                
                # Read encrypted data
                encrypted_data = data[offset:offset+length]
                offset += length + 1  # +1 for newline
                
                try:
                    # Decrypt and parse
                    decrypted_data = self.fernet.decrypt(encrypted_data)
                    event_data = json.loads(decrypted_data.decode())
                    event = AuditEvent.from_dict(event_data)
                    
                    # Filter by time and type
                    if start_time <= event.timestamp <= end_time:
                        if event_types is None or event.event_type in event_types:
                            events.append(event)
                
                except Exception as e:
                    logging.error(f"Error decrypting audit event: {e}")
        
        except Exception as e:
            logging.error(f"Error reading encrypted audit file {file_path}: {e}")
        
        return events


class AuditLogger:
    """Main audit logging system"""
    
    def __init__(self, storage: AuditStorage, secret_key: Optional[str] = None,
                 min_level: AuditLevel = AuditLevel.INFO, 
                 async_logging: bool = True):
        self.storage = storage
        self.secret_key = secret_key or "default_audit_secret_key"
        self.min_level = min_level
        self.async_logging = async_logging
        
        # Async logging setup
        if async_logging:
            self.event_queue = queue.Queue(maxsize=10000)
            self.executor = ThreadPoolExecutor(max_workers=2)
            self.logging_thread = threading.Thread(target=self._async_logging_loop, daemon=True)
            self.running = True
            self.logging_thread.start()
        
        # Session tracking
        self.current_session_id = str(uuid.uuid4())
        self.current_user_id = None
        
        # Statistics
        self.stats = {
            'events_logged': 0,
            'events_dropped': 0,
            'integrity_violations': 0,
            'signature_failures': 0
        }
        
        self.lock = threading.Lock()
    
    def set_session_context(self, session_id: str, user_id: Optional[str] = None):
        """Set current session context"""
        self.current_session_id = session_id
        self.current_user_id = user_id
    
    def log_event(self, event_type: AuditEventType, message: str,
                  level: AuditLevel = AuditLevel.INFO,
                  source: str = "quantum_system",
                  details: Optional[Dict[str, Any]] = None,
                  user_id: Optional[str] = None,
                  session_id: Optional[str] = None,
                  ip_address: Optional[str] = None,
                  user_agent: Optional[str] = None,
                  request_id: Optional[str] = None):
        """Log an audit event"""
        
        # Check minimum level
        if level.value < self.min_level.value:
            return
        
        # Create event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            event_type=event_type,
            timestamp=time.time(),
            level=level,
            source=source,
            user_id=user_id or self.current_user_id,
            session_id=session_id or self.current_session_id,
            message=message,
            details=details or {},
            ip_address=ip_address,
            user_agent=user_agent,
            request_id=request_id
        )
        
        # Sign event
        event.sign(self.secret_key)
        
        # Log event
        if self.async_logging:
            try:
                self.event_queue.put_nowait(event)
            except queue.Full:
                with self.lock:
                    self.stats['events_dropped'] += 1
        else:
            self._write_event(event)
    
    def _async_logging_loop(self):
        """Async logging loop"""
        while self.running:
            try:
                event = self.event_queue.get(timeout=1.0)
                self._write_event(event)
                self.event_queue.task_done()
            except queue.Empty:
                continue
            except Exception as e:
                logging.error(f"Error in async logging loop: {e}")
    
    def _write_event(self, event: AuditEvent):
        """Write event to storage"""
        try:
            self.storage.write_event(event)
            with self.lock:
                self.stats['events_logged'] += 1
        except Exception as e:
            logging.error(f"Error writing audit event: {e}")
            with self.lock:
                self.stats['events_dropped'] += 1
    
    def get_events(self, start_time: float, end_time: float,
                  event_types: Optional[List[AuditEventType]] = None,
                  verify_integrity: bool = True) -> List[AuditEvent]:
        """Get audit events from storage"""
        events = self.storage.read_events(start_time, end_time, event_types)
        
        if verify_integrity:
            verified_events = []
            for event in events:
                if event.verify_integrity() and event.verify_signature(self.secret_key):
                    verified_events.append(event)
                else:
                    with self.lock:
                        if not event.verify_integrity():
                            self.stats['integrity_violations'] += 1
                        if not event.verify_signature(self.secret_key):
                            self.stats['signature_failures'] += 1
            
            return verified_events
        
        return events
    
    def search_events(self, query: str, start_time: float, end_time: float,
                     case_sensitive: bool = False) -> List[AuditEvent]:
        """Search audit events by message content"""
        events = self.get_events(start_time, end_time)
        
        if not case_sensitive:
            query = query.lower()
        
        matching_events = []
        for event in events:
            search_text = event.message
            if not case_sensitive:
                search_text = search_text.lower()
            
            if query in search_text:
                matching_events.append(event)
        
        return matching_events
    
    def generate_compliance_report(self, start_time: float, end_time: float) -> Dict[str, Any]:
        """Generate compliance report"""
        events = self.get_events(start_time, end_time)
        
        # Categorize events
        event_categories = {}
        security_events = []
        error_events = []
        access_events = []
        
        for event in events:
            # Count by type
            event_type = event.event_type.value
            event_categories[event_type] = event_categories.get(event_type, 0) + 1
            
            # Categorize by security relevance
            if event.level == AuditLevel.SECURITY or 'security' in event.event_type.value:
                security_events.append(event)
            
            if event.level in [AuditLevel.ERROR, AuditLevel.CRITICAL]:
                error_events.append(event)
            
            if 'access' in event.event_type.value or 'auth' in event.event_type.value:
                access_events.append(event)
        
        # Generate statistics
        total_events = len(events)
        unique_users = len(set(event.user_id for event in events if event.user_id))
        unique_sessions = len(set(event.session_id for event in events if event.session_id))
        
        return {
            'report_period': {
                'start_time': start_time,
                'end_time': end_time,
                'duration_hours': (end_time - start_time) / 3600
            },
            'summary': {
                'total_events': total_events,
                'unique_users': unique_users,
                'unique_sessions': unique_sessions,
                'security_events': len(security_events),
                'error_events': len(error_events),
                'access_events': len(access_events)
            },
            'event_categories': event_categories,
            'integrity_status': {
                'events_logged': self.stats['events_logged'],
                'events_dropped': self.stats['events_dropped'],
                'integrity_violations': self.stats['integrity_violations'],
                'signature_failures': self.stats['signature_failures']
            },
            'top_error_messages': self._get_top_error_messages(error_events),
            'security_incidents': [event.to_dict() for event in security_events[:10]],
            'compliance_issues': self._identify_compliance_issues(events)
        }
    
    def _get_top_error_messages(self, error_events: List[AuditEvent], limit: int = 5) -> List[Dict[str, Any]]:
        """Get most common error messages"""
        error_counts = {}
        
        for event in error_events:
            message = event.message
            error_counts[message] = error_counts.get(message, 0) + 1
        
        # Sort by count and return top N
        sorted_errors = sorted(error_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [
            {'message': message, 'count': count}
            for message, count in sorted_errors[:limit]
        ]
    
    def _identify_compliance_issues(self, events: List[AuditEvent]) -> List[str]:
        """Identify potential compliance issues"""
        issues = []
        
        # Check for failed authentications
        failed_auths = [e for e in events if e.event_type == AuditEventType.ACCESS_DENIED]
        if len(failed_auths) > 10:
            issues.append(f"High number of failed authentication attempts: {len(failed_auths)}")
        
        # Check for security violations
        security_violations = [e for e in events if e.event_type == AuditEventType.SECURITY_VIOLATION]
        if security_violations:
            issues.append(f"Security violations detected: {len(security_violations)}")
        
        # Check for data access patterns
        data_access_events = [e for e in events if e.event_type == AuditEventType.DATA_ACCESS]
        if len(data_access_events) > 1000:
            issues.append(f"High volume of data access events: {len(data_access_events)}")
        
        # Check for configuration changes
        config_changes = [e for e in events if e.event_type == AuditEventType.CONFIGURATION_CHANGED]
        if len(config_changes) > 5:
            issues.append(f"Multiple configuration changes: {len(config_changes)}")
        
        return issues
    
    def cleanup_old_events(self, retention_days: int = 90):
        """Clean up old audit events"""
        cutoff_time = time.time() - (retention_days * 24 * 3600)
        self.storage.archive_events(cutoff_time)
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get audit logging statistics"""
        with self.lock:
            return self.stats.copy()
    
    def stop(self):
        """Stop async logging"""
        if self.async_logging:
            self.running = False
            if self.logging_thread:
                self.logging_thread.join(timeout=5.0)
            self.executor.shutdown(wait=True)


# Convenience functions for common audit events
def log_task_submitted(audit_logger: AuditLogger, task_id: str, user_id: str, task_details: Dict[str, Any]):
    """Log task submission"""
    audit_logger.log_event(
        event_type=AuditEventType.TASK_SUBMITTED,
        message=f"Task {task_id} submitted",
        level=AuditLevel.INFO,
        source="task_engine",
        details={
            'task_id': task_id,
            'task_details': task_details
        },
        user_id=user_id
    )


def log_security_violation(audit_logger: AuditLogger, violation_type: str, details: Dict[str, Any], 
                          ip_address: Optional[str] = None):
    """Log security violation"""
    audit_logger.log_event(
        event_type=AuditEventType.SECURITY_VIOLATION,
        message=f"Security violation: {violation_type}",
        level=AuditLevel.SECURITY,
        source="security_monitor",
        details=details,
        ip_address=ip_address
    )


def log_resource_allocation(audit_logger: AuditLogger, task_id: str, resources: Dict[str, Any]):
    """Log resource allocation"""
    audit_logger.log_event(
        event_type=AuditEventType.RESOURCE_ALLOCATED,
        message=f"Resources allocated for task {task_id}",
        level=AuditLevel.INFO,
        source="resource_allocator",
        details={
            'task_id': task_id,
            'resources': resources
        }
    )


if __name__ == "__main__":
    # Example usage and testing
    print("Quantum Audit Logging System")
    print("=" * 50)
    
    # Create storage and audit logger
    storage = FileAuditStorage("/tmp/quantum_audit_test")
    audit_logger = AuditLogger(storage, secret_key="test_secret_key")
    
    try:
        print("\nLogging test events...")
        
        # Set session context
        audit_logger.set_session_context("test_session_123", "test_user")
        
        # Log various types of events
        audit_logger.log_event(
            AuditEventType.SYSTEM_START,
            "Quantum system started",
            AuditLevel.INFO,
            details={'version': '1.0.0', 'config': 'production'}
        )
        
        log_task_submitted(audit_logger, "task_001", "user_123", {
            'name': 'Test Task',
            'priority': 'high'
        })
        
        log_security_violation(audit_logger, "Invalid access attempt", {
            'attempted_resource': '/admin/config',
            'user_agent': 'curl/7.68.0'
        }, ip_address="192.168.1.100")
        
        log_resource_allocation(audit_logger, "task_001", {
            'cpu': 2.0,
            'memory': 1024
        })
        
        audit_logger.log_event(
            AuditEventType.TASK_COMPLETED,
            "Task completed successfully",
            AuditLevel.INFO,
            details={'task_id': 'task_001', 'duration': 125.5}
        )
        
        # Wait for async logging to complete
        time.sleep(1)
        
        print("\nRetrieving and verifying events...")
        
        # Get events from last hour
        end_time = time.time()
        start_time = end_time - 3600
        
        events = audit_logger.get_events(start_time, end_time)
        print(f"Retrieved {len(events)} events")
        
        for event in events:
            print(f"  {event.event_type.value}: {event.message}")
            print(f"    Integrity: {'✓' if event.verify_integrity() else '✗'}")
            print(f"    Signature: {'✓' if event.verify_signature(audit_logger.secret_key) else '✗'}")
        
        # Search events
        print("\nSearching for task-related events...")
        task_events = audit_logger.search_events("task", start_time, end_time)
        print(f"Found {len(task_events)} task-related events")
        
        # Generate compliance report
        print("\nGenerating compliance report...")
        report = audit_logger.generate_compliance_report(start_time, end_time)
        
        print(f"Compliance Report Summary:")
        print(f"  Total Events: {report['summary']['total_events']}")
        print(f"  Security Events: {report['summary']['security_events']}")
        print(f"  Error Events: {report['summary']['error_events']}")
        print(f"  Unique Users: {report['summary']['unique_users']}")
        
        if report['compliance_issues']:
            print(f"  Compliance Issues:")
            for issue in report['compliance_issues']:
                print(f"    - {issue}")
        
        # Test encrypted storage
        print("\nTesting encrypted audit storage...")
        encrypted_storage = EncryptedAuditStorage("/tmp/quantum_audit_encrypted_test")
        encrypted_logger = AuditLogger(encrypted_storage, secret_key="test_secret_key")
        
        encrypted_logger.log_event(
            AuditEventType.DATA_ACCESS,
            "Sensitive data accessed",
            AuditLevel.SECURITY,
            details={'data_type': 'user_records', 'record_count': 100}
        )
        
        time.sleep(0.5)
        
        encrypted_events = encrypted_logger.get_events(time.time() - 60, time.time())
        print(f"Encrypted storage test: {len(encrypted_events)} events stored and retrieved")
        
        # Show statistics
        stats = audit_logger.get_statistics()
        print(f"\nAudit Logger Statistics:")
        print(f"  Events Logged: {stats['events_logged']}")
        print(f"  Events Dropped: {stats['events_dropped']}")
        print(f"  Integrity Violations: {stats['integrity_violations']}")
        print(f"  Signature Failures: {stats['signature_failures']}")
        
    finally:
        audit_logger.stop()
        encrypted_logger.stop()
    
    print("\nAudit logging demonstration completed!")
