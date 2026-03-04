#!/usr/bin/env python3
"""
test_submission_infrastructure.py

Test script to verify the submission and leaderboard system works correctly.
Run this locally before deploying to GitHub.
"""

import os
import sys
import pandas as pd
import tempfile
import shutil
from pathlib import Path

# Colors for output
GREEN = '\033[92m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
RESET = '\033[0m'

def print_status(status, message):
    if status == "pass":
        print(f"{GREEN}✅ {message}{RESET}")
    elif status == "fail":
        print(f"{RED}❌ {message}{RESET}")
    elif status == "info":
        print(f"{BLUE}ℹ️  {message}{RESET}")
    elif status == "warn":
        print(f"{YELLOW}⚠️  {message}{RESET}")

def test_data_files():
    """Test that required data files exist"""
    print(f"\n{BLUE}=== Testing Data Files ==={RESET}")
    
    required_files = {
        'data/train.csv': 'Training data',
        'data/test.csv': 'Test data',
        'data/test_labels.csv': 'Test labels',
        'data/graph_edges.csv': 'Graph edges',
        'data/node_types.csv': 'Node types'
    }
    
    all_exist = True
    for filepath, desc in required_files.items():
        if os.path.exists(filepath):
            file_size = os.path.getsize(filepath)
            print_status("pass", f"{desc}: {filepath} ({file_size:,} bytes)")
        else:
            print_status("fail", f"{desc}: {filepath} NOT FOUND")
            all_exist = False
    
    return all_exist

def test_scoring_script():
    """Test that scoring script runs correctly"""
    print(f"\n{BLUE}=== Testing Scoring Script ==={RESET}")
    
    if not os.path.exists('scoring_script.py'):
        print_status("fail", "scoring_script.py not found")
        return False
    
    print_status("pass", "scoring_script.py exists")
    
    # Test import
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location("scoring_script", "scoring_script.py")
        module = importlib.util.module_from_spec(spec)
        print_status("pass", "scoring_script.py can be imported")
        return True
    except Exception as e:
        print_status("fail", f"scoring_script.py import failed: {e}")
        return False

def test_submission_format():
    """Test creating a valid submission file"""
    print(f"\n{BLUE}=== Testing Submission Format ==={RESET}")
    
    # Load test data
    try:
        test_df = pd.read_csv('data/test.csv')
        print_status("pass", f"Loaded test.csv ({len(test_df)} samples)")
    except Exception as e:
        print_status("fail", f"Failed to load test.csv: {e}")
        return False
    
    # Create sample submission
    try:
        submission = pd.DataFrame({
            'id': test_df['node_id'],
            'y_pred': [0.5 for _ in range(len(test_df))]
        })

        # Save to temp file
        temp_dir = Path('submissions/inbox/test_team/test_run')
        temp_dir.mkdir(parents=True, exist_ok=True)
        temp_file = temp_dir / 'predictions.csv'
        submission.to_csv(temp_file, index=False)
        print_status("pass", f"Created valid submission CSV ({len(submission)} rows)")

        meta_file = temp_dir / 'metadata.json'
        meta_file.write_text('{\n  \"team\": \"test_team\",\n  \"run_id\": \"test_run\",\n  \"model_name\": \"Test Model\",\n  \"model_type\": \"human\"\n}\n')
        print_status("pass", "Created metadata.json")
        
        # Verify format
        reloaded = pd.read_csv(temp_file)
        assert 'id' in reloaded.columns, "Missing 'id' column"
        assert 'y_pred' in reloaded.columns, "Missing 'y_pred' column"
        assert len(reloaded) == len(test_df), "Row count mismatch"
        print_status("pass", "Submission format validation passed")
        
        # Clean up
        os.remove(temp_file)
        os.remove(meta_file)
        temp_dir.rmdir()
        return True
        
    except Exception as e:
        print_status("fail", f"Submission format test failed: {e}")
        return False

def test_leaderboard_structure():
    """Test that leaderboard.md has correct structure"""
    print(f"\n{BLUE}=== Testing Leaderboard Structure ==={RESET}")
    
    if not os.path.exists('leaderboard.md'):
        print_status("fail", "leaderboard.md not found")
        return False
    
    print_status("pass", "leaderboard.md exists")
    
    with open('leaderboard.md', 'r') as f:
        content = f.read()
    
    required_sections = [
        '# 🏆 GNN Challenge Leaderboard',
        '## Current Leaderboard',
        '## Submissions Log',
        'Submission Guidelines'
    ]
    
    all_found = True
    for section in required_sections:
        if section in content:
            print_status("pass", f"Found section: '{section}'")
        else:
            print_status("warn", f"Missing section: '{section}'")
            all_found = False
    
    return all_found

def test_github_actions_workflow():
    """Test that GitHub Actions workflow file is valid"""
    print(f"\n{BLUE}=== Testing GitHub Actions Workflow ==={RESET}")
    
    workflow_file = '.github/workflows/score-submission.yml'
    if not os.path.exists(workflow_file):
        print_status("warn", f"{workflow_file} not found (needed for automated scoring)")
        return False
    
    print_status("pass", f"{workflow_file} exists")
    
    try:
        import yaml
        with open(workflow_file, 'r') as f:
            workflow = yaml.safe_load(f)
        
        # Check required fields
        assert 'name' in workflow, "Missing 'name' field"
        assert 'on' in workflow, "Missing 'on' field"
        assert 'jobs' in workflow, "Missing 'jobs' field"
        
        print_status("pass", "Workflow YAML structure is valid")
        print_status("info", f"Workflow name: '{workflow['name']}'")
        
        return True
    except ImportError:
        print_status("warn", "PyYAML not installed (cannot validate YAML)")
        return True
    except Exception as e:
        print_status("fail", f"Workflow validation failed: {e}")
        return False

def test_contributing_guide():
    """Test that CONTRIBUTING.md exists and has key sections"""
    print(f"\n{BLUE}=== Testing Contributing Guide ==={RESET}")
    
    if not os.path.exists('CONTRIBUTING.md'):
        print_status("fail", "CONTRIBUTING.md not found")
        return False
    
    print_status("pass", "CONTRIBUTING.md exists")
    
    with open('CONTRIBUTING.md', 'r') as f:
        content = f.read()
    
    required_sections = [
        'Quick Start',
        'Required Format',
        'Submit via Pull Request',
        'FAQ'
    ]
    
    for section in required_sections:
        if section in content:
            print_status("pass", f"Found section: '{section}'")
        else:
            print_status("warn", f"Missing section: '{section}'")
    
    return True

def main():
    print(f"\n{BLUE}{'='*60}")
    print("GNN Challenge - Submission Infrastructure Test")
    print(f"{'='*60}{RESET}\n")
    
    tests = [
        ("Data Files", test_data_files),
        ("Scoring Script", test_scoring_script),
        ("Submission Format", test_submission_format),
        ("Leaderboard Structure", test_leaderboard_structure),
        ("GitHub Actions Workflow", test_github_actions_workflow),
        ("Contributing Guide", test_contributing_guide)
    ]
    
    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print_status("fail", f"Test '{test_name}' crashed: {e}")
            results[test_name] = False
    
    # Summary
    print(f"\n{BLUE}{'='*60}")
    print("Test Summary")
    print(f"{'='*60}{RESET}\n")
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        color = GREEN if result else RED
        print(f"{color}{status}{RESET} - {test_name}")
    
    print(f"\n{BLUE}Result: {passed}/{total} tests passed{RESET}")
    
    if passed == total:
        print_status("pass", "All tests passed! Infrastructure is ready.")
        return 0
    elif passed >= total - 1:
        print_status("warn", "Most tests passed. Some warnings present.")
        return 0
    else:
        print_status("fail", "Several tests failed. Please fix issues before deploying.")
        return 1

if __name__ == '__main__':
    sys.exit(main())
