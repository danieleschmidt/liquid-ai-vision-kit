#!/usr/bin/env python3
"""
Task Validation and Security Framework
Comprehensive validation, sanitization, and security checks for quantum tasks
"""

import re
import hashlib
import hmac
import time
import json
import logging
from typing import Dict, List, Optional, Set, Any, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import numpy as np
from pathlib import Path


class ValidationResult(Enum):
    """Validation result types"""
    VALID = "valid"
    WARNING = "warning"
    ERROR = "error"
    BLOCKED = "blocked"


class SecurityLevel(Enum):
    """Security levels for tasks"""
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    RESTRICTED = 3
    TOP_SECRET = 4


@dataclass
class ValidationIssue:
    """Validation issue details"""
    level: ValidationResult
    code: str
    message: str
    field: Optional[str] = None
    suggested_fix: Optional[str] = None
    security_risk: bool = False


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    is_valid: bool
    security_level: SecurityLevel
    issues: List[ValidationIssue]
    warnings: List[str]
    recommendations: List[str]
    risk_score: float  # 0.0 to 1.0
    validation_time: float
    
    def has_errors(self) -> bool:
        return any(issue.level in [ValidationResult.ERROR, ValidationResult.BLOCKED] for issue in self.issues)
    
    def has_warnings(self) -> bool:
        return any(issue.level == ValidationResult.WARNING for issue in self.issues)
    
    def get_security_risks(self) -> List[ValidationIssue]:
        return [issue for issue in self.issues if issue.security_risk]


class TaskValidator:
    """Comprehensive task validation and security framework"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Security patterns
        self.dangerous_patterns = {
            'command_injection': [
                r'[;&|`$()]',
                r'\\x[0-9a-fA-F]{2}',
                r'eval\s*\(',
                r'exec\s*\(',
                r'__import__\s*\(',
                r'subprocess\.',
                r'os\.system',
                r'shell=True'
            ],
            'path_traversal': [
                r'\.\./+',
                r'\.\.\\+',
                r'/etc/',
                r'/proc/',
                r'/dev/',
                r'C:\\Windows',
                r'%SYSTEMROOT%'
            ],
            'sql_injection': [
                r'(\b(SELECT|INSERT|UPDATE|DELETE|DROP|CREATE|ALTER)\b.*?;)',
                r'(\bUNION\b.*?\bSELECT\b)',
                r'(\b(OR|AND)\s+\d+\s*=\s*\d+)',
                r'(\'\s*(OR|AND)\s+\'\w*\'\s*=\s*\'\w*\')'
            ],
            'xss': [
                r'<script[^>]*>.*?</script>',
                r'javascript:',
                r'onload\s*=',
                r'onerror\s*=',
                r'onclick\s*='
            ]
        }
        
        # Resource limits
        self.resource_limits = {
            'max_memory': 16 * 1024 * 1024 * 1024,  # 16GB
            'max_cpu_cores': 64,
            'max_disk_space': 1024 * 1024 * 1024 * 1024,  # 1TB
            'max_network_bandwidth': 10 * 1024 * 1024 * 1024,  # 10Gbps
            'max_execution_time': 24 * 3600,  # 24 hours
            'max_dependencies': 100,
            'max_task_name_length': 256,
            'max_metadata_size': 1024 * 1024  # 1MB
        }
        
        # Allowed task patterns
        self.allowed_patterns = {
            'task_id': r'^[a-zA-Z0-9_\-]{1,64}$',
            'task_name': r'^[a-zA-Z0-9\s_\-\.]{1,256}$',
            'resource_name': r'^[a-zA-Z0-9_]{1,32}$'
        }
        
        # Blocked domains and IPs
        self.blocked_domains = {
            'suspicious_domains.txt',
            'malware_domains.txt'
        }
        
        self.blocked_ips = set()
        
        # Load security rules
        self._load_security_rules()
    
    def _load_security_rules(self):
        """Load additional security rules from configuration"""
        try:
            # This would load from config files in production
            pass
        except Exception as e:
            self.logger.warning(f"Could not load additional security rules: {e}")
    
    def validate_task(self, task_data: Dict[str, Any]) -> ValidationReport:
        """Comprehensive task validation"""
        start_time = time.time()
        issues = []
        warnings = []
        recommendations = []
        risk_score = 0.0
        
        try:
            # Basic structure validation
            structure_issues = self._validate_structure(task_data)
            issues.extend(structure_issues)
            
            # Security validation
            security_issues, security_risk = self._validate_security(task_data)
            issues.extend(security_issues)
            risk_score += security_risk
            
            # Resource validation
            resource_issues, resource_risk = self._validate_resources(task_data)
            issues.extend(resource_issues)
            risk_score += resource_risk
            
            # Dependency validation
            dependency_issues = self._validate_dependencies(task_data)
            issues.extend(dependency_issues)
            
            # Content validation
            content_issues = self._validate_content(task_data)
            issues.extend(content_issues)
            
            # Performance validation
            perf_warnings, perf_recommendations = self._validate_performance(task_data)
            warnings.extend(perf_warnings)
            recommendations.extend(perf_recommendations)
            
            # Determine security level
            security_level = self._determine_security_level(task_data, risk_score)
            
            # Final validation
            is_valid = not any(issue.level in [ValidationResult.ERROR, ValidationResult.BLOCKED] 
                             for issue in issues)
            
        except Exception as e:
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                code="VALIDATION_ERROR",
                message=f"Validation process failed: {str(e)}",
                security_risk=True
            ))
            is_valid = False
            security_level = SecurityLevel.RESTRICTED
            risk_score = 1.0
        
        validation_time = time.time() - start_time
        
        return ValidationReport(
            is_valid=is_valid,
            security_level=security_level,
            issues=issues,
            warnings=warnings,
            recommendations=recommendations,
            risk_score=min(1.0, risk_score),
            validation_time=validation_time
        )
    
    def _validate_structure(self, task_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate task data structure"""
        issues = []
        
        # Required fields
        required_fields = ['id', 'name', 'priority']
        for field in required_fields:
            if field not in task_data:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="MISSING_FIELD",
                    message=f"Required field '{field}' is missing",
                    field=field,
                    suggested_fix=f"Add '{field}' field to task data"
                ))
        
        # Field type validation
        field_types = {
            'id': str,
            'name': str,
            'priority': (int, str),
            'estimated_duration': (int, float),
            'max_retries': int,
            'dependencies': list
        }
        
        for field, expected_type in field_types.items():
            if field in task_data:
                if not isinstance(task_data[field], expected_type):
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="INVALID_TYPE",
                        message=f"Field '{field}' must be of type {expected_type.__name__}",
                        field=field,
                        suggested_fix=f"Convert '{field}' to {expected_type.__name__}"
                    ))
        
        # Pattern validation
        for field, pattern in self.allowed_patterns.items():
            if field in task_data and isinstance(task_data[field], str):
                if not re.match(pattern, task_data[field]):
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="INVALID_FORMAT",
                        message=f"Field '{field}' does not match required format",
                        field=field,
                        suggested_fix=f"Ensure '{field}' matches pattern: {pattern}"
                    ))
        
        return issues
    
    def _validate_security(self, task_data: Dict[str, Any]) -> Tuple[List[ValidationIssue], float]:
        """Validate task security"""
        issues = []
        risk_score = 0.0
        
        # Check for dangerous patterns
        for field, value in task_data.items():
            if isinstance(value, str):
                for category, patterns in self.dangerous_patterns.items():
                    for pattern in patterns:
                        if re.search(pattern, value, re.IGNORECASE):
                            issues.append(ValidationIssue(
                                level=ValidationResult.BLOCKED,
                                code=f"SECURITY_RISK_{category.upper()}",
                                message=f"Potential {category} detected in field '{field}'",
                                field=field,
                                security_risk=True,
                                suggested_fix="Remove or sanitize dangerous content"
                            ))
                            risk_score += 0.3
        
        # Check callback security
        if 'callback' in task_data:
            callback_issues, callback_risk = self._validate_callback_security(task_data['callback'])
            issues.extend(callback_issues)
            risk_score += callback_risk
        
        # Check metadata security
        if 'metadata' in task_data:
            metadata_issues, metadata_risk = self._validate_metadata_security(task_data['metadata'])
            issues.extend(metadata_issues)
            risk_score += metadata_risk
        
        return issues, risk_score
    
    def _validate_callback_security(self, callback: Any) -> Tuple[List[ValidationIssue], float]:
        """Validate callback function security"""
        issues = []
        risk_score = 0.0
        
        if callable(callback):
            # Check function name
            func_name = getattr(callback, '__name__', 'unknown')
            if any(dangerous in func_name.lower() for dangerous in ['eval', 'exec', 'compile', 'open']):
                issues.append(ValidationIssue(
                    level=ValidationResult.BLOCKED,
                    code="DANGEROUS_CALLBACK",
                    message=f"Callback function '{func_name}' may be dangerous",
                    security_risk=True,
                    suggested_fix="Use safer callback functions"
                ))
                risk_score += 0.5
            
            # Check for access to dangerous modules
            if hasattr(callback, '__globals__'):
                globals_dict = callback.__globals__
                dangerous_modules = {'os', 'subprocess', 'sys', 'eval', 'exec'}
                for module in dangerous_modules:
                    if module in globals_dict:
                        issues.append(ValidationIssue(
                            level=ValidationResult.WARNING,
                            code="CALLBACK_MODULE_ACCESS",
                            message=f"Callback has access to potentially dangerous module: {module}",
                            security_risk=True,
                            suggested_fix="Limit callback access to safe modules only"
                        ))
                        risk_score += 0.2
        
        elif isinstance(callback, str):
            # Check for code injection in string callbacks
            for pattern in self.dangerous_patterns['command_injection']:
                if re.search(pattern, callback):
                    issues.append(ValidationIssue(
                        level=ValidationResult.BLOCKED,
                        code="CODE_INJECTION_CALLBACK",
                        message="Potential code injection in callback string",
                        security_risk=True,
                        suggested_fix="Sanitize or use function objects instead of strings"
                    ))
                    risk_score += 0.4
        
        return issues, risk_score
    
    def _validate_metadata_security(self, metadata: Dict[str, Any]) -> Tuple[List[ValidationIssue], float]:
        """Validate metadata security"""
        issues = []
        risk_score = 0.0
        
        # Check metadata size
        metadata_size = len(json.dumps(metadata))
        if metadata_size > self.resource_limits['max_metadata_size']:
            issues.append(ValidationIssue(
                level=ValidationResult.ERROR,
                code="METADATA_TOO_LARGE",
                message=f"Metadata size ({metadata_size} bytes) exceeds limit ({self.resource_limits['max_metadata_size']} bytes)",
                suggested_fix="Reduce metadata size"
            ))
            risk_score += 0.2
        
        # Check for sensitive information
        sensitive_keys = {'password', 'key', 'secret', 'token', 'credential', 'api_key'}
        for key, value in metadata.items():
            key_lower = key.lower()
            if any(sensitive in key_lower for sensitive in sensitive_keys):
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="SENSITIVE_METADATA",
                    message=f"Potentially sensitive information in metadata key: {key}",
                    field=f"metadata.{key}",
                    security_risk=True,
                    suggested_fix="Remove sensitive information or encrypt it"
                ))
                risk_score += 0.1
        
        return issues, risk_score
    
    def _validate_resources(self, task_data: Dict[str, Any]) -> Tuple[List[ValidationIssue], float]:
        """Validate resource requirements"""
        issues = []
        risk_score = 0.0
        
        if 'resources' in task_data:
            resources = task_data['resources']
            
            for resource_name, amount in resources.items():
                # Check resource name format
                if not re.match(self.allowed_patterns['resource_name'], resource_name):
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="INVALID_RESOURCE_NAME",
                        message=f"Invalid resource name: {resource_name}",
                        field=f"resources.{resource_name}",
                        suggested_fix="Use valid resource name format"
                    ))
                
                # Check resource limits
                if resource_name == 'memory' and amount > self.resource_limits['max_memory']:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="EXCESSIVE_MEMORY",
                        message=f"Memory requirement ({amount}) exceeds limit ({self.resource_limits['max_memory']})",
                        field=f"resources.{resource_name}",
                        suggested_fix="Reduce memory requirement"
                    ))
                    risk_score += 0.3
                
                elif resource_name == 'cpu' and amount > self.resource_limits['max_cpu_cores']:
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="EXCESSIVE_CPU",
                        message=f"CPU requirement ({amount}) exceeds limit ({self.resource_limits['max_cpu_cores']})",
                        field=f"resources.{resource_name}",
                        suggested_fix="Reduce CPU requirement"
                    ))
                    risk_score += 0.3
        
        # Check execution time limits
        if 'estimated_duration' in task_data:
            duration = task_data['estimated_duration']
            if duration > self.resource_limits['max_execution_time']:
                issues.append(ValidationIssue(
                    level=ValidationResult.WARNING,
                    code="LONG_EXECUTION",
                    message=f"Estimated duration ({duration}s) is very long",
                    field="estimated_duration",
                    suggested_fix="Consider breaking into smaller tasks"
                ))
                risk_score += 0.1
        
        return issues, risk_score
    
    def _validate_dependencies(self, task_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate task dependencies"""
        issues = []
        
        if 'dependencies' in task_data:
            dependencies = task_data['dependencies']
            
            # Check dependency count
            if len(dependencies) > self.resource_limits['max_dependencies']:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="TOO_MANY_DEPENDENCIES",
                    message=f"Task has {len(dependencies)} dependencies (max: {self.resource_limits['max_dependencies']})",
                    field="dependencies",
                    suggested_fix="Reduce number of dependencies"
                ))
            
            # Check for circular dependencies (basic check)
            task_id = task_data.get('id')
            if task_id and task_id in dependencies:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="CIRCULAR_DEPENDENCY",
                    message="Task depends on itself",
                    field="dependencies",
                    suggested_fix="Remove self-dependency"
                ))
            
            # Validate dependency format
            for dep in dependencies:
                if not isinstance(dep, str) or not re.match(self.allowed_patterns['task_id'], dep):
                    issues.append(ValidationIssue(
                        level=ValidationResult.ERROR,
                        code="INVALID_DEPENDENCY_FORMAT",
                        message=f"Invalid dependency format: {dep}",
                        field="dependencies",
                        suggested_fix="Use valid task ID format for dependencies"
                    ))
        
        return issues
    
    def _validate_content(self, task_data: Dict[str, Any]) -> List[ValidationIssue]:
        """Validate task content"""
        issues = []
        
        # Name length check
        if 'name' in task_data:
            name = task_data['name']
            if len(name) > self.resource_limits['max_task_name_length']:
                issues.append(ValidationIssue(
                    level=ValidationResult.ERROR,
                    code="NAME_TOO_LONG",
                    message=f"Task name too long ({len(name)} chars, max: {self.resource_limits['max_task_name_length']})",
                    field="name",
                    suggested_fix="Shorten task name"
                ))
        
        # Check for empty values
        empty_fields = []
        for field, value in task_data.items():
            if isinstance(value, str) and not value.strip():
                empty_fields.append(field)
            elif isinstance(value, (list, dict)) and not value:
                empty_fields.append(field)
        
        if empty_fields:
            issues.append(ValidationIssue(
                level=ValidationResult.WARNING,
                code="EMPTY_FIELDS",
                message=f"Empty fields detected: {', '.join(empty_fields)}",
                suggested_fix="Provide values for empty fields or remove them"
            ))
        
        return issues
    
    def _validate_performance(self, task_data: Dict[str, Any]) -> Tuple[List[str], List[str]]:
        """Validate performance-related aspects"""
        warnings = []
        recommendations = []
        
        # Check for potential performance issues
        if 'estimated_duration' in task_data:
            duration = task_data['estimated_duration']
            if duration > 3600:  # 1 hour
                warnings.append(f"Long running task ({duration/3600:.1f} hours)")
                recommendations.append("Consider implementing progress reporting")
        
        # Resource efficiency checks
        if 'resources' in task_data:
            resources = task_data['resources']
            
            # Check for resource imbalance
            if 'cpu' in resources and 'memory' in resources:
                cpu_cores = resources['cpu']
                memory_gb = resources['memory'] / 1024 / 1024 / 1024
                ratio = memory_gb / cpu_cores
                
                if ratio < 1:  # Less than 1GB per core
                    recommendations.append("Consider increasing memory allocation for CPU-intensive tasks")
                elif ratio > 8:  # More than 8GB per core
                    recommendations.append("Memory allocation may be excessive for CPU requirements")
        
        return warnings, recommendations
    
    def _determine_security_level(self, task_data: Dict[str, Any], risk_score: float) -> SecurityLevel:
        """Determine appropriate security level"""
        if risk_score >= 0.8:
            return SecurityLevel.TOP_SECRET
        elif risk_score >= 0.6:
            return SecurityLevel.RESTRICTED
        elif risk_score >= 0.4:
            return SecurityLevel.CONFIDENTIAL
        elif risk_score >= 0.2:
            return SecurityLevel.INTERNAL
        else:
            return SecurityLevel.PUBLIC
    
    def sanitize_task_data(self, task_data: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize task data by removing or fixing issues"""
        sanitized = task_data.copy()
        
        # Remove dangerous patterns from strings
        for field, value in sanitized.items():
            if isinstance(value, str):
                # Remove potentially dangerous characters
                for patterns in self.dangerous_patterns.values():
                    for pattern in patterns:
                        value = re.sub(pattern, '', value, flags=re.IGNORECASE)
                sanitized[field] = value.strip()
        
        # Limit resource requirements
        if 'resources' in sanitized:
            resources = sanitized['resources']
            if 'memory' in resources:
                resources['memory'] = min(resources['memory'], self.resource_limits['max_memory'])
            if 'cpu' in resources:
                resources['cpu'] = min(resources['cpu'], self.resource_limits['max_cpu_cores'])
        
        # Limit execution time
        if 'estimated_duration' in sanitized:
            sanitized['estimated_duration'] = min(
                sanitized['estimated_duration'],
                self.resource_limits['max_execution_time']
            )
        
        # Limit dependencies
        if 'dependencies' in sanitized:
            deps = sanitized['dependencies'][:self.resource_limits['max_dependencies']]
            sanitized['dependencies'] = [dep for dep in deps if re.match(self.allowed_patterns['task_id'], str(dep))]
        
        return sanitized
    
    def generate_task_hash(self, task_data: Dict[str, Any], secret_key: str = "") -> str:
        """Generate cryptographic hash for task integrity"""
        task_json = json.dumps(task_data, sort_keys=True, separators=(',', ':'))
        if secret_key:
            return hmac.new(secret_key.encode(), task_json.encode(), hashlib.sha256).hexdigest()
        else:
            return hashlib.sha256(task_json.encode()).hexdigest()
    
    def verify_task_integrity(self, task_data: Dict[str, Any], expected_hash: str, secret_key: str = "") -> bool:
        """Verify task data integrity"""
        computed_hash = self.generate_task_hash(task_data, secret_key)
        return hmac.compare_digest(computed_hash, expected_hash)


if __name__ == "__main__":
    # Example usage and testing
    print("Task Validation and Security Framework")
    print("=" * 50)
    
    validator = TaskValidator()
    
    # Test with various task examples
    test_tasks = [
        # Valid task
        {
            'id': 'test_task_001',
            'name': 'Data Processing',
            'priority': 1,
            'resources': {'cpu': 2, 'memory': 1024},
            'estimated_duration': 300,
            'dependencies': ['setup_task'],
            'metadata': {'version': '1.0', 'author': 'system'}
        },
        
        # Task with security issues
        {
            'id': 'suspicious_task',
            'name': 'Suspicious Task; rm -rf /',
            'priority': 1,
            'resources': {'cpu': 100, 'memory': 32 * 1024 * 1024 * 1024},
            'callback': 'eval("malicious_code()")',
            'metadata': {'api_key': 'secret123', 'password': 'hidden'}
        },
        
        # Task with validation errors
        {
            'name': '',  # Missing ID
            'priority': 'invalid',  # Wrong type
            'resources': {'invalid@name': 100},  # Invalid resource name
            'dependencies': ['task1'] * 150  # Too many dependencies
        }
    ]
    
    for i, task in enumerate(test_tasks, 1):
        print(f"\nValidating Task {i}:")
        print("=" * 30)
        
        report = validator.validate_task(task)
        
        print(f"Valid: {report.is_valid}")
        print(f"Security Level: {report.security_level.name}")
        print(f"Risk Score: {report.risk_score:.2f}")
        print(f"Validation Time: {report.validation_time*1000:.2f}ms")
        
        if report.issues:
            print(f"\nIssues ({len(report.issues)}):")
            for issue in report.issues:
                print(f"  [{issue.level.name}] {issue.code}: {issue.message}")
                if issue.suggested_fix:
                    print(f"    Fix: {issue.suggested_fix}")
        
        if report.warnings:
            print(f"\nWarnings ({len(report.warnings)}):")
            for warning in report.warnings:
                print(f"  - {warning}")
        
        if report.recommendations:
            print(f"\nRecommendations ({len(report.recommendations)}):")
            for rec in report.recommendations:
                print(f"  - {rec}")
        
        # Test sanitization
        if not report.is_valid:
            print(f"\nSanitized Task:")
            sanitized = validator.sanitize_task_data(task)
            sanitized_report = validator.validate_task(sanitized)
            print(f"  Valid after sanitization: {sanitized_report.is_valid}")
            print(f"  New risk score: {sanitized_report.risk_score:.2f}")
        
        # Test integrity
        task_hash = validator.generate_task_hash(task, "secret_key")
        print(f"\nTask Hash: {task_hash[:16]}...")
        
        integrity_valid = validator.verify_task_integrity(task, task_hash, "secret_key")
        print(f"Integrity Check: {'PASS' if integrity_valid else 'FAIL'}")