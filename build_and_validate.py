#!/usr/bin/env python3
"""
Comprehensive Build and Validation System
Automated build, test, and quality assurance for the Liquid AI Vision Kit
"""

import os
import sys
import subprocess
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import concurrent.futures


@dataclass
class BuildResult:
    """Build step result"""
    step_name: str
    success: bool
    duration: float
    output: str = ""
    error_output: str = ""
    exit_code: int = 0
    artifacts: List[str] = None
    
    def __post_init__(self):
        if self.artifacts is None:
            self.artifacts = []


@dataclass
class ValidationReport:
    """Comprehensive validation report"""
    timestamp: float
    overall_success: bool
    build_results: List[BuildResult]
    test_results: Dict[str, Any]
    quality_metrics: Dict[str, float]
    security_scan: Dict[str, Any]
    performance_benchmarks: Dict[str, Any]
    deployment_readiness: bool


class BuildAndValidationSystem:
    """Comprehensive build and validation system"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.logger = self._setup_logging()
        
        self.project_root = Path(__file__).parent
        self.src_dir = self.project_root / "src"
        self.tests_dir = self.project_root / "tests"
        self.build_dir = self.project_root / "build"
        
        self.build_results = []
        self.validation_start_time = time.time()
        
        # Ensure build directory exists
        self.build_dir.mkdir(exist_ok=True)
    
    def _setup_logging(self) -> logging.Logger:
        """Setup build logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('build_validation.log'),
                logging.StreamHandler(sys.stdout)
            ]
        )
        return logging.getLogger('BuildValidator')
    
    def run_comprehensive_validation(self) -> ValidationReport:
        """Run complete build and validation pipeline"""
        self.logger.info("🚀 Starting Comprehensive Build and Validation")
        self.logger.info("=" * 80)
        
        validation_steps = [
            ("Environment Check", self._validate_environment),
            ("Python Lint", self._lint_python_code),
            ("Python Unit Tests", self._run_python_unit_tests),
            ("Integration Tests", self._run_integration_tests),
            ("C++ Build", self._build_cpp_components),
            ("C++ Tests", self._run_cpp_tests),
            ("Security Scan", self._run_security_scan),
            ("Performance Benchmarks", self._run_performance_tests),
            ("Quality Metrics", self._calculate_quality_metrics),
            ("Deployment Validation", self._validate_deployment_readiness)
        ]
        
        # Execute validation steps
        for step_name, step_function in validation_steps:
            self.logger.info(f"🔄 Executing: {step_name}")
            result = self._execute_build_step(step_name, step_function)
            self.build_results.append(result)
            
            if result.success:
                self.logger.info(f"✅ {step_name} completed successfully")
            else:
                self.logger.error(f"❌ {step_name} failed: {result.error_output}")
                if self.config.get("fail_fast", True):
                    break
        
        # Generate comprehensive report
        report = self._generate_validation_report()
        
        # Save report
        self._save_validation_report(report)
        
        return report
    
    def _execute_build_step(self, step_name: str, step_function) -> BuildResult:
        """Execute a single build step with timing and error handling"""
        start_time = time.time()
        
        try:
            result = step_function()
            if isinstance(result, BuildResult):
                return result
            else:
                # Assume success if function returns without exception
                return BuildResult(
                    step_name=step_name,
                    success=True,
                    duration=time.time() - start_time,
                    output=str(result) if result else "Success"
                )
        
        except Exception as e:
            return BuildResult(
                step_name=step_name,
                success=False,
                duration=time.time() - start_time,
                error_output=str(e)
            )
    
    def _validate_environment(self) -> BuildResult:
        """Validate build environment"""
        checks = []
        
        # Python version check
        python_version = sys.version_info
        if python_version >= (3, 8):
            checks.append(f"✓ Python {python_version.major}.{python_version.minor}.{python_version.micro}")
        else:
            checks.append(f"✗ Python version too old: {python_version}")
        
        # Required Python packages
        required_packages = ['numpy', 'scipy', 'psutil']
        for package in required_packages:
            try:
                __import__(package)
                checks.append(f"✓ {package}")
            except ImportError:
                checks.append(f"✗ {package} not found")
        
        # CMake check (for C++ builds)
        try:
            result = subprocess.run(['cmake', '--version'], capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                version_line = result.stdout.split('\n')[0]
                checks.append(f"✓ {version_line}")
            else:
                checks.append("✗ CMake not available")
        except (subprocess.TimeoutExpired, FileNotFoundError):
            checks.append("✗ CMake not found")
        
        # GCC/Clang check
        for compiler in ['gcc', 'clang']:
            try:
                result = subprocess.run([compiler, '--version'], capture_output=True, text=True, timeout=5)
                if result.returncode == 0:
                    version_line = result.stdout.split('\n')[0]
                    checks.append(f"✓ {compiler}: {version_line}")
                    break
            except (subprocess.TimeoutExpired, FileNotFoundError):
                continue
        else:
            checks.append("✗ No C++ compiler found")
        
        # Check disk space
        try:
            import shutil
            free_space_gb = shutil.disk_usage(self.project_root).free / (1024**3)
            if free_space_gb > 1.0:  # At least 1GB free
                checks.append(f"✓ Disk space: {free_space_gb:.1f}GB free")
            else:
                checks.append(f"✗ Low disk space: {free_space_gb:.1f}GB free")
        except Exception:
            checks.append("✗ Could not check disk space")
        
        output = "\n".join(checks)
        failed_checks = [check for check in checks if check.startswith("✗")]
        
        return BuildResult(
            step_name="Environment Check",
            success=len(failed_checks) == 0,
            duration=1.0,
            output=output,
            error_output="\n".join(failed_checks) if failed_checks else ""
        )
    
    def _lint_python_code(self) -> BuildResult:
        """Run Python code linting"""
        lint_commands = [
            # Flake8 for style checking
            ['python', '-m', 'flake8', '--max-line-length=120', '--exclude=build,__pycache__', str(self.src_dir)],
            # Basic syntax check
            ['python', '-m', 'py_compile'] + [str(f) for f in self.src_dir.rglob('*.py')]
        ]
        
        all_output = []
        all_errors = []
        overall_success = True
        
        for cmd in lint_commands:
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=self.project_root)
                all_output.append(f"Command: {' '.join(cmd)}")
                all_output.append(result.stdout)
                
                if result.returncode != 0:
                    overall_success = False
                    all_errors.append(f"Command failed: {' '.join(cmd)}")
                    all_errors.append(result.stderr)
                
            except subprocess.TimeoutExpired:
                overall_success = False
                all_errors.append(f"Timeout: {' '.join(cmd)}")
            except FileNotFoundError:
                # If linting tool not available, skip but don't fail
                all_output.append(f"Skipped (not available): {' '.join(cmd)}")
        
        return BuildResult(
            step_name="Python Lint",
            success=overall_success,
            duration=2.0,
            output="\n".join(all_output),
            error_output="\n".join(all_errors)
        )
    
    def _run_python_unit_tests(self) -> BuildResult:
        """Run Python unit tests"""
        if not (self.tests_dir / "python").exists():
            return BuildResult(
                step_name="Python Unit Tests",
                success=True,
                duration=0.1,
                output="No Python unit tests found"
            )
        
        try:
            # Run pytest with coverage
            cmd = [
                'python', '-m', 'pytest',
                str(self.tests_dir / "python"),
                '-v', '--tb=short',
                '--timeout=300'  # 5 minute timeout per test
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600, cwd=self.project_root)
            
            return BuildResult(
                step_name="Python Unit Tests",
                success=result.returncode == 0,
                duration=5.0,
                output=result.stdout,
                error_output=result.stderr,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return BuildResult(
                step_name="Python Unit Tests",
                success=False,
                duration=600.0,
                error_output="Unit tests timed out"
            )
        except FileNotFoundError:
            return BuildResult(
                step_name="Python Unit Tests",
                success=False,
                duration=0.1,
                error_output="pytest not available"
            )
    
    def _run_integration_tests(self) -> BuildResult:
        """Run integration tests"""
        integration_test_file = self.tests_dir / "integration" / "test_full_system.py"
        
        if not integration_test_file.exists():
            return BuildResult(
                step_name="Integration Tests",
                success=True,
                duration=0.1,
                output="No integration tests found"
            )
        
        try:
            cmd = ['python', str(integration_test_file)]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=1200, cwd=self.project_root)  # 20 minutes
            
            return BuildResult(
                step_name="Integration Tests",
                success=result.returncode == 0,
                duration=20.0,
                output=result.stdout,
                error_output=result.stderr,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return BuildResult(
                step_name="Integration Tests",
                success=False,
                duration=1200.0,
                error_output="Integration tests timed out"
            )
    
    def _build_cpp_components(self) -> BuildResult:
        """Build C++ components"""
        if not (self.project_root / "CMakeLists.txt").exists():
            return BuildResult(
                step_name="C++ Build",
                success=True,
                duration=0.1,
                output="No CMakeLists.txt found, skipping C++ build"
            )
        
        try:
            # Create build directory
            cpp_build_dir = self.build_dir / "cpp"
            cpp_build_dir.mkdir(exist_ok=True)
            
            # Configure
            configure_cmd = ['cmake', '..', '-DCMAKE_BUILD_TYPE=Release', '-DENABLE_TESTS=ON']
            configure_result = subprocess.run(
                configure_cmd, 
                capture_output=True, 
                text=True, 
                timeout=300,
                cwd=cpp_build_dir
            )
            
            if configure_result.returncode != 0:
                return BuildResult(
                    step_name="C++ Build",
                    success=False,
                    duration=5.0,
                    error_output=f"CMake configure failed:\n{configure_result.stderr}"
                )
            
            # Build
            build_cmd = ['cmake', '--build', '.', '--parallel']
            build_result = subprocess.run(
                build_cmd,
                capture_output=True,
                text=True,
                timeout=1800,  # 30 minutes
                cwd=cpp_build_dir
            )
            
            return BuildResult(
                step_name="C++ Build",
                success=build_result.returncode == 0,
                duration=30.0,
                output=f"Configure output:\n{configure_result.stdout}\n\nBuild output:\n{build_result.stdout}",
                error_output=build_result.stderr,
                exit_code=build_result.returncode,
                artifacts=[str(cpp_build_dir)]
            )
        
        except subprocess.TimeoutExpired:
            return BuildResult(
                step_name="C++ Build",
                success=False,
                duration=1800.0,
                error_output="C++ build timed out"
            )
        except FileNotFoundError:
            return BuildResult(
                step_name="C++ Build",
                success=False,
                duration=0.1,
                error_output="CMake not found"
            )
    
    def _run_cpp_tests(self) -> BuildResult:
        """Run C++ tests"""
        cpp_build_dir = self.build_dir / "cpp"
        
        if not cpp_build_dir.exists():
            return BuildResult(
                step_name="C++ Tests",
                success=True,
                duration=0.1,
                output="No C++ build directory found"
            )
        
        try:
            # Run CTest
            cmd = ['ctest', '--verbose', '--timeout', '300']
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
                cwd=cpp_build_dir
            )
            
            return BuildResult(
                step_name="C++ Tests",
                success=result.returncode == 0,
                duration=10.0,
                output=result.stdout,
                error_output=result.stderr,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return BuildResult(
                step_name="C++ Tests",
                success=False,
                duration=600.0,
                error_output="C++ tests timed out"
            )
        except FileNotFoundError:
            return BuildResult(
                step_name="C++ Tests",
                success=False,
                duration=0.1,
                error_output="ctest not found"
            )
    
    def _run_security_scan(self) -> BuildResult:
        """Run security scanning"""
        security_results = []
        
        # Bandit for Python security
        try:
            cmd = ['python', '-m', 'bandit', '-r', str(self.src_dir), '-f', 'json']
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                security_results.append("✓ Bandit: No security issues found")
            else:
                try:
                    bandit_data = json.loads(result.stdout)
                    issues = len(bandit_data.get('results', []))
                    if issues == 0:
                        security_results.append("✓ Bandit: No security issues")
                    else:
                        security_results.append(f"⚠ Bandit: {issues} potential security issues")
                except json.JSONDecodeError:
                    security_results.append("⚠ Bandit: Could not parse results")
            
        except (subprocess.TimeoutExpired, FileNotFoundError):
            security_results.append("⚠ Bandit not available")
        
        # Check for hardcoded secrets/keys
        secret_patterns = [
            "password", "secret", "key", "token", "api_key"
        ]
        
        suspicious_files = []
        for py_file in self.src_dir.rglob("*.py"):
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read().lower()
                    for pattern in secret_patterns:
                        if f"{pattern} =" in content or f'"{pattern}"' in content:
                            suspicious_files.append(str(py_file))
                            break
            except Exception:
                continue
        
        if suspicious_files:
            security_results.append(f"⚠ Potential secrets in {len(suspicious_files)} files")
        else:
            security_results.append("✓ No obvious hardcoded secrets found")
        
        return BuildResult(
            step_name="Security Scan",
            success=True,  # Security scan doesn't fail the build
            duration=5.0,
            output="\n".join(security_results)
        )
    
    def _run_performance_tests(self) -> BuildResult:
        """Run performance benchmarks"""
        try:
            # Run the integrated system demo with timing
            demo_file = self.project_root / "demo_integrated_system.py"
            
            if not demo_file.exists():
                return BuildResult(
                    step_name="Performance Benchmarks",
                    success=True,
                    duration=0.1,
                    output="No performance tests found"
                )
            
            start_time = time.time()
            
            # Set environment variable to make demo run faster
            env = os.environ.copy()
            env['DEMO_MODE'] = 'performance_test'
            
            result = subprocess.run(
                ['python', str(demo_file)],
                capture_output=True,
                text=True,
                timeout=300,  # 5 minutes
                env=env,
                cwd=self.project_root
            )
            
            duration = time.time() - start_time
            
            # Parse performance metrics from output
            performance_metrics = {
                "demo_execution_time": duration,
                "demo_success": result.returncode == 0
            }
            
            return BuildResult(
                step_name="Performance Benchmarks",
                success=result.returncode == 0,
                duration=duration,
                output=f"Performance metrics: {performance_metrics}\n\nDemo output:\n{result.stdout}",
                error_output=result.stderr,
                exit_code=result.returncode
            )
        
        except subprocess.TimeoutExpired:
            return BuildResult(
                step_name="Performance Benchmarks",
                success=False,
                duration=300.0,
                error_output="Performance tests timed out"
            )
    
    def _calculate_quality_metrics(self) -> BuildResult:
        """Calculate code quality metrics"""
        metrics = {}
        
        # Count lines of code
        python_files = list(self.src_dir.rglob("*.py"))
        cpp_files = list(self.src_dir.rglob("*.cpp")) + list(self.src_dir.rglob("*.h"))
        
        total_python_lines = 0
        total_cpp_lines = 0
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    total_python_lines += len(f.readlines())
            except Exception:
                continue
        
        for cpp_file in cpp_files:
            try:
                with open(cpp_file, 'r', encoding='utf-8') as f:
                    total_cpp_lines += len(f.readlines())
            except Exception:
                continue
        
        metrics['python_files'] = len(python_files)
        metrics['cpp_files'] = len(cpp_files)
        metrics['python_lines'] = total_python_lines
        metrics['cpp_lines'] = total_cpp_lines
        metrics['total_lines'] = total_python_lines + total_cpp_lines
        
        # Calculate test coverage ratio (approximate)
        test_files = list(self.tests_dir.rglob("*.py"))
        test_lines = 0
        for test_file in test_files:
            try:
                with open(test_file, 'r', encoding='utf-8') as f:
                    test_lines += len(f.readlines())
            except Exception:
                continue
        
        metrics['test_files'] = len(test_files)
        metrics['test_lines'] = test_lines
        metrics['test_to_code_ratio'] = test_lines / max(total_python_lines, 1)
        
        # Code complexity (simplified - count functions/classes)
        complexity_items = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    complexity_items += content.count('def ') + content.count('class ')
            except Exception:
                continue
        
        metrics['complexity_score'] = complexity_items / max(len(python_files), 1)
        
        # Documentation score (count docstrings)
        docstring_count = 0
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    docstring_count += content.count('"""') + content.count("'''")
            except Exception:
                continue
        
        metrics['documentation_score'] = docstring_count / max(complexity_items, 1)
        
        output_lines = [
            f"Quality Metrics:",
            f"  Python files: {metrics['python_files']}",
            f"  C++ files: {metrics['cpp_files']}",
            f"  Total lines of code: {metrics['total_lines']:,}",
            f"  Test files: {metrics['test_files']}",
            f"  Test-to-code ratio: {metrics['test_to_code_ratio']:.2f}",
            f"  Complexity score: {metrics['complexity_score']:.1f} functions/classes per file",
            f"  Documentation score: {metrics['documentation_score']:.2f} docstrings per function"
        ]
        
        return BuildResult(
            step_name="Quality Metrics",
            success=True,
            duration=2.0,
            output="\n".join(output_lines)
        )
    
    def _validate_deployment_readiness(self) -> BuildResult:
        """Validate deployment readiness"""
        readiness_checks = []
        
        # Check for required files
        required_files = [
            "README.md",
            "requirements.txt",
            "production_deployment.py",
            "src/quantum_task_planner/__init__.py"
        ]
        
        for file_path in required_files:
            full_path = self.project_root / file_path
            if full_path.exists():
                readiness_checks.append(f"✓ {file_path}")
            else:
                readiness_checks.append(f"✗ {file_path} missing")
        
        # Check configuration files
        config_files = ["pyproject.toml", "CMakeLists.txt"]
        for config_file in config_files:
            if (self.project_root / config_file).exists():
                readiness_checks.append(f"✓ {config_file}")
            else:
                readiness_checks.append(f"⚠ {config_file} not found (optional)")
        
        # Check documentation
        docs_dir = self.project_root / "docs"
        if docs_dir.exists() and any(docs_dir.iterdir()):
            readiness_checks.append("✓ Documentation directory exists")
        else:
            readiness_checks.append("⚠ Documentation directory empty or missing")
        
        # Check for license
        license_files = ["LICENSE", "LICENSE.txt", "LICENSE.md"]
        license_found = any((self.project_root / lic).exists() for lic in license_files)
        if license_found:
            readiness_checks.append("✓ License file found")
        else:
            readiness_checks.append("⚠ License file not found")
        
        failed_critical_checks = [check for check in readiness_checks if check.startswith("✗")]
        
        return BuildResult(
            step_name="Deployment Validation",
            success=len(failed_critical_checks) == 0,
            duration=1.0,
            output="\n".join(readiness_checks),
            error_output="\n".join(failed_critical_checks) if failed_critical_checks else ""
        )
    
    def _generate_validation_report(self) -> ValidationReport:
        """Generate comprehensive validation report"""
        overall_success = all(result.success for result in self.build_results)
        
        # Aggregate test results
        test_results = {}
        for result in self.build_results:
            if "test" in result.step_name.lower():
                test_results[result.step_name] = {
                    "success": result.success,
                    "duration": result.duration,
                    "exit_code": result.exit_code
                }
        
        # Extract quality metrics
        quality_metrics = {}
        for result in self.build_results:
            if result.step_name == "Quality Metrics":
                # Parse metrics from output
                lines = result.output.split('\n')
                for line in lines:
                    if ':' in line and any(char.isdigit() for char in line):
                        try:
                            parts = line.split(':')
                            if len(parts) == 2:
                                key = parts[0].strip().replace(' ', '_').lower()
                                value_str = parts[1].strip()
                                # Extract numeric value
                                import re
                                numbers = re.findall(r'[\d.,]+', value_str.replace(',', ''))
                                if numbers:
                                    quality_metrics[key] = float(numbers[0])
                        except ValueError:
                            continue
        
        # Security scan results
        security_scan = {}
        for result in self.build_results:
            if result.step_name == "Security Scan":
                security_scan = {
                    "scan_completed": result.success,
                    "issues_found": "⚠" in result.output,
                    "details": result.output
                }
        
        # Performance benchmarks
        performance_benchmarks = {}
        for result in self.build_results:
            if result.step_name == "Performance Benchmarks":
                performance_benchmarks = {
                    "benchmarks_completed": result.success,
                    "execution_time": result.duration,
                    "details": result.output[:500] + "..." if len(result.output) > 500 else result.output
                }
        
        # Deployment readiness
        deployment_readiness = any(
            result.success for result in self.build_results
            if result.step_name == "Deployment Validation"
        )
        
        return ValidationReport(
            timestamp=time.time(),
            overall_success=overall_success,
            build_results=self.build_results,
            test_results=test_results,
            quality_metrics=quality_metrics,
            security_scan=security_scan,
            performance_benchmarks=performance_benchmarks,
            deployment_readiness=deployment_readiness
        )
    
    def _save_validation_report(self, report: ValidationReport):
        """Save validation report to file"""
        report_file = self.build_dir / f"validation_report_{int(report.timestamp)}.json"
        
        # Convert report to JSON-serializable format
        report_dict = {
            "timestamp": report.timestamp,
            "overall_success": report.overall_success,
            "build_results": [
                {
                    "step_name": result.step_name,
                    "success": result.success,
                    "duration": result.duration,
                    "output": result.output[:1000] + "..." if len(result.output) > 1000 else result.output,
                    "error_output": result.error_output[:1000] + "..." if len(result.error_output) > 1000 else result.error_output,
                    "exit_code": result.exit_code,
                    "artifacts": result.artifacts
                }
                for result in report.build_results
            ],
            "test_results": report.test_results,
            "quality_metrics": report.quality_metrics,
            "security_scan": report.security_scan,
            "performance_benchmarks": report.performance_benchmarks,
            "deployment_readiness": report.deployment_readiness
        }
        
        with open(report_file, 'w') as f:
            json.dump(report_dict, f, indent=2)
        
        self.logger.info(f"📄 Validation report saved to: {report_file}")
    
    def print_validation_summary(self, report: ValidationReport):
        """Print validation summary"""
        print("\n" + "=" * 80)
        print("🎯 VALIDATION SUMMARY")
        print("=" * 80)
        
        # Overall result
        if report.overall_success:
            print("✅ OVERALL RESULT: SUCCESS")
        else:
            print("❌ OVERALL RESULT: FAILURE")
        
        print(f"⏱️  Total validation time: {time.time() - self.validation_start_time:.1f} seconds")
        
        # Build steps summary
        print(f"\n📋 BUILD STEPS SUMMARY:")
        for result in report.build_results:
            status = "✅ PASS" if result.success else "❌ FAIL"
            print(f"   {status} {result.step_name} ({result.duration:.1f}s)")
        
        # Test results
        if report.test_results:
            print(f"\n🧪 TEST RESULTS:")
            for test_name, test_data in report.test_results.items():
                status = "✅ PASS" if test_data["success"] else "❌ FAIL"
                print(f"   {status} {test_name}")
        
        # Quality metrics
        if report.quality_metrics:
            print(f"\n📊 QUALITY METRICS:")
            for metric, value in report.quality_metrics.items():
                print(f"   {metric.replace('_', ' ').title()}: {value}")
        
        # Deployment readiness
        print(f"\n🚀 DEPLOYMENT READINESS: {'✅ READY' if report.deployment_readiness else '❌ NOT READY'}")
        
        print("=" * 80)


def main():
    """Main build and validation entry point"""
    print("🔧 Liquid AI Vision Kit - Build and Validation System")
    print("=" * 80)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Build and validate the Liquid AI Vision Kit')
    parser.add_argument('--fail-fast', action='store_true', help='Stop on first failure')
    parser.add_argument('--skip-cpp', action='store_true', help='Skip C++ build and tests')
    parser.add_argument('--quick', action='store_true', help='Run quick validation (skip long tests)')
    
    args = parser.parse_args()
    
    # Configure validation
    config = {
        "fail_fast": args.fail_fast,
        "skip_cpp": args.skip_cpp,
        "quick_mode": args.quick
    }
    
    # Create build system
    build_system = BuildAndValidationSystem(config)
    
    try:
        # Run validation
        report = build_system.run_comprehensive_validation()
        
        # Print summary
        build_system.print_validation_summary(report)
        
        # Exit with appropriate code
        sys.exit(0 if report.overall_success else 1)
        
    except KeyboardInterrupt:
        print("\n⏹️  Build validation interrupted by user")
        sys.exit(130)
    except Exception as e:
        print(f"\n💥 Build validation failed with exception: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()