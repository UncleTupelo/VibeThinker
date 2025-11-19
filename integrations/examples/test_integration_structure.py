"""
Simple test to validate the integration structure

This test validates that the integration files are properly structured
and can be imported without errors (doesn't require model download).
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


def test_imports():
    """Test that all integration files can be imported"""
    print("Testing imports...")
    
    try:
        # Test that files exist and are valid Python
        import ast
        
        files_to_test = [
            "../openai_api_server.py",
            "chatgpt_compatible_client.py",
            "claude_style_prompts.py",
            "perplexity_style_search.py",
        ]
        
        for file in files_to_test:
            filepath = os.path.join(os.path.dirname(__file__), file)
            print(f"  Checking {file}...")
            
            with open(filepath, 'r') as f:
                code = f.read()
                ast.parse(code)  # Check if valid Python syntax
            
            print(f"  ✓ {file} is valid Python")
        
        print("\n✓ All integration files are valid!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_api_structure():
    """Test that API server has required endpoints"""
    print("\nTesting API server structure...")
    
    try:
        import ast
        filepath = os.path.join(os.path.dirname(__file__), "../openai_api_server.py")
        
        with open(filepath, 'r') as f:
            code = f.read()
        
        # Check for required components
        required_strings = [
            "FastAPI",
            "/v1/chat/completions",
            "/v1/models",
            "ChatCompletionRequest",
            "ChatCompletionResponse",
        ]
        
        for req in required_strings:
            if req in code:
                print(f"  ✓ Found {req}")
            else:
                print(f"  ✗ Missing {req}")
                return False
        
        print("\n✓ API server has required structure!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def test_documentation():
    """Test that documentation files exist and are non-empty"""
    print("\nTesting documentation...")
    
    try:
        docs = [
            "../README.md",
            "../requirements.txt",
        ]
        
        for doc in docs:
            filepath = os.path.join(os.path.dirname(__file__), doc)
            print(f"  Checking {doc}...")
            
            if not os.path.exists(filepath):
                print(f"  ✗ {doc} does not exist")
                return False
            
            with open(filepath, 'r') as f:
                content = f.read()
                if len(content) < 100:
                    print(f"  ✗ {doc} seems too short")
                    return False
            
            print(f"  ✓ {doc} exists and has content")
        
        print("\n✓ All documentation is present!")
        return True
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 80)
    print("VibeThinker Integration Validation Test")
    print("=" * 80)
    print()
    
    results = []
    
    results.append(("Import Test", test_imports()))
    results.append(("API Structure Test", test_api_structure()))
    results.append(("Documentation Test", test_documentation()))
    
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    all_passed = all(result for _, result in results)
    
    print()
    if all_passed:
        print("✓ All tests passed! Integration is properly structured.")
        return 0
    else:
        print("✗ Some tests failed. Please check the output above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
