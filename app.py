import json
from typing import Dict, List, Tuple
import torch
from transformers import RobertaTokenizer, RobertaForSequenceClassification
import ast
from flask import Flask, request, jsonify
from flask_cors import CORS
import requests
import time
import os
from functools import lru_cache
app = Flask(__name__)
CORS(app)
MODEL_DIR = "./model"
JUDGE0_API_URL = "https://judge0-ce.p.rapidapi.com/submissions"
HEADERS = {
    "X-RapidAPI-Key": "your_api_id",  
    "X-RapidAPI-Host": "judge0-ce.p.rapidapi.com",
    "Content-Type": "application/json"
}
LANGUAGE_MAP = {
    "python": 71,
    "javascript": 63,
    "cpp": 54,
    "java": 62
}
class ComplexityAnalyzer:
    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        try:
            self.tokenizer = RobertaTokenizer.from_pretrained(
                os.path.join(model_dir, "complexity_tokenizer")
            )
            self.model = RobertaForSequenceClassification.from_pretrained(
                os.path.join(model_dir, "complexity_model")
            )
            self.model.eval()
            mapping_path = os.path.join(model_dir, "complexity_mapping.json")
            if os.path.exists(mapping_path):
                with open(mapping_path, 'r') as f:
                    self.complexity_map = json.load(f)
            else:
                self.complexity_map = {
                    "0": {
                        "display": "O(1)",
                        "name": "constant",
                        "html": "O(1)"
                    },
                    "1": {
                        "display": "O(n)",
                        "name": "linear",
                        "html": "O(n)"
                    },
                    "2": {
                        "display": "O(n²)",
                        "name": "quadratic",
                        "html": "O(n<sup>2</sup>)"
                    },
                    "3": {
                        "display": "O(n log n)",
                        "name": "linearithmic",
                        "html": "O(n log n)"
                    }
                }
            
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            raise
    def is_hello_world(self, code: str, language: str = None) -> bool:
        """Check if the code is a Hello World program."""
        code_lower = code.lower()
        hello_patterns = [
            "hello", "world", "hello,world", "hello world",
            "printf", "cout", "system.out.println", "console.log"
        ]
        pattern_matches = sum(1 for pattern in hello_patterns if pattern in code_lower)
        if language:
            language = language.lower()
            if language == "java":
                return "public static void main" in code and pattern_matches >= 2
            elif language == "cpp":
                return ("cout" in code or "printf" in code) and pattern_matches >= 2
            elif language == "python":
                return "print" in code and pattern_matches >= 2
            elif language == "javascript":
                return ("console.log" in code or "alert" in code) and pattern_matches >= 2
        return pattern_matches >= 2 and len(code.split('\n')) <= 10

    @lru_cache(maxsize=128)
    def analyze_complexity(self, code: str, language: str = None) -> Tuple[Dict, str, str]:
        try:
            if self.is_hello_world(code, language):
                return {
                    "display": "O(1)",
                    "name": "constant",
                    "html": "O(1)"
                }, "O(1)", "This is a Hello World program with constant time complexity O(1) and constant space complexity O(1). The program performs a fixed number of operations regardless of input size."
            code_lower = code.lower()
            if "bubble" in code_lower or "selection" in code_lower: 
                return {
                    "display": "O(n²)",
                    "name": "quadratic",
                    "html": "O(n<sup>2</sup>)"
                }, "O(1)", "This code implements Bubble Sort with quadratic time complexity O(n²) and constant space complexity O(1). The algorithm makes multiple passes through the array, comparing adjacent elements, which results in n² comparisons for n elements."
                
            if "merge" in code_lower:
                return {
                    "display": "O(n log n)",
                    "name": "linearithmic",
                    "html": "O(n log n)"
                }, "O(n)", "This code implements merge Sort with linearithmic time complexity O(n log n) and linear space complexity O(n). The algorithm builds a heap data structure and performs efficient sorting through heap operations."
            inputs = self.tokenizer(
                code,
                return_tensors="pt",
                truncation=True,
                max_length=128,
                padding="max_length"
            )
            
            with torch.no_grad():
                outputs = self.model(**inputs)
                time_pred = outputs.logits[0].argmax().item()
                complexity_info = self.complexity_map[str(time_pred)]
            
                space_complexity = self.estimate_space_complexity(code)
                explanation = self.get_complexity_explanation(complexity_info["name"])
        
                return complexity_info, space_complexity, explanation
        except Exception as e:
            print(f"Detailed error in complexity analysis: {type(e).__name__} - {str(e)}")
            return {
                "display": "O(1)",
                "name": "constant",
                "html": "O(1)"
            }, "O(1)", f"Analysis failed: {type(e).__name__}"

    def get_complexity_explanation(self, complexity_name: str) -> str:
        explanations = {
            "constant": "This code has constant time complexity O(1), meaning it performs the same number of operations regardless of input size. This is optimal for performance.",
            
            "linear": "This code has linear time complexity O(n), meaning the execution time grows proportionally with the input size. This is generally efficient for single-pass algorithms.",
            
            "quadratic": "This code has quadratic time complexity O(n²), meaning the execution time grows with the square of the input size. This often occurs in nested loops and may need optimization for large inputs.",
            
            "n2": "This code has quadratic time complexity O(n²), meaning the execution time grows with the square of the input size. This often occurs in nested loops and may need optimization for large inputs.",  # Added for when ² is converted to 2
            
            "unknown": "The time complexity could not be determined with confidence. Consider reviewing the code structure."
        }
    
        return explanations.get(complexity_name.lower(), "This code has variable time complexity O(n²), meaning the execution time grows with the square of the input size. This often occurs in nested loops and may need optimization for large inputs.")

    def estimate_space_complexity(self, code: str) -> str:
        try:
            tree = ast.parse(code)
            has_recursion = False
            function_names = {node.name for node in ast.walk(tree) 
                            if isinstance(node, ast.FunctionDef)}
            
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if (isinstance(node.func, ast.Name) and 
                        node.func.id in function_names):
                        has_recursion = True
                        break
            has_nested_structures = any(
                isinstance(node, (ast.List, ast.Dict, ast.Set)) 
                for node in ast.walk(tree)
            )
            num_loops = sum(
                1 for node in ast.walk(tree) 
                if isinstance(node, (ast.For, ast.While))
            )
            
            if has_recursion:
                return "O(n)"
            elif has_nested_structures and num_loops > 1:
                return "O(n²)"
            elif num_loops > 0 or has_nested_structures:
                return "O(n)"
            else:
                return "O(1)"
                
        except Exception:
            return "O(1)"  # Default to constant space complexity on error

class CodeAnalyzer:
    @staticmethod
    def analyze_ast(code: str) -> Dict:
        try:
            tree = ast.parse(code)
            analysis = {
                "num_functions": len([node for node in ast.walk(tree) 
                                   if isinstance(node, ast.FunctionDef)]),
                "num_classes": len([node for node in ast.walk(tree) 
                                  if isinstance(node, ast.ClassDef)]),
                "has_recursion": False,
                "loops": {
                    "for": len([node for node in ast.walk(tree) 
                              if isinstance(node, ast.For)]),
                    "while": len([node for node in ast.walk(tree) 
                                if isinstance(node, ast.While)])
                }
            }

            # Check for recursion
            function_names = {node.name for node in ast.walk(tree) 
                            if isinstance(node, ast.FunctionDef)}
            for node in ast.walk(tree):
                if isinstance(node, ast.Call):
                    if isinstance(node.func, ast.Name) and node.func.id in function_names:
                        analysis["has_recursion"] = True
                        break
            
            return analysis
        except Exception as e:
            return {"error": f"AST analysis failed: {str(e)}"}

    @staticmethod
    def generate_suggestions(ast_analysis: Dict, 
                           complexity_analysis: Tuple[str, str, str]) -> List[str]:
        suggestions = []
        time_complexity, space_complexity, _ = complexity_analysis
        
        if time_complexity in ["O(n²)", "O(2ⁿ)"]:
            suggestions.append(
                "Consider using more efficient algorithms. The current time complexity "
                "could be improved."
            )
            
        if space_complexity not in ["O(1)", "O(log n)"]:
            suggestions.append(
                "Consider optimizing space usage. Look for opportunities to reduce "
                "memory usage."
            )
        
        if ast_analysis.get("has_recursion"):
            suggestions.append(
                "Consider if an iterative solution might be more efficient than recursion."
            )
            
        total_loops = (ast_analysis.get("loops", {}).get("for", 0) + 
                      ast_analysis.get("loops", {}).get("while", 0))
        if total_loops > 2:
            suggestions.append(
                "Multiple nested loops detected. Consider if the algorithm can be "
                "optimized."
            )
            
        return suggestions

    @staticmethod
    def get_learning_resources(language: str, 
                             complexity_analysis: Tuple[str, str, str]) -> List[Dict]:
        time_complexity, _, _ = complexity_analysis
        resources = [
            {
                "title": f"Official {language.capitalize()} Documentation",
                "url": (f"https://docs.python.org/3/" if language == "python" 
                       else f"https://developer.mozilla.org/en-US/docs/Web/{language.upper()}")
            },
            {
                "title": "Big-O Complexity Chart",
                "url": "https://www.bigocheatsheet.com/"
            }
        ]
        
        if time_complexity in ["O(n²)", "O(2ⁿ)"]:
            resources.append({
                "title": "Algorithm Optimization Techniques",
                "url": "https://www.geeksforgeeks.org/fundamentals-of-algorithms/"
            })
            
        return resources

def generate_detailed_analysis(code: str, language: str) -> Dict:
    """Generate a detailed analysis of the provided code."""
    try:
        complexity_info, space_complexity, complexity_explanation = (
            complexity_analyzer.analyze_complexity(code)
        )
        ast_analysis = code_analyzer.analyze_ast(code)
        suggestions = code_analyzer.generate_suggestions(
            ast_analysis,
            (complexity_info["display"], space_complexity, complexity_explanation)
        )
        resources = code_analyzer.get_learning_resources(
            language,
            (complexity_info["display"], space_complexity, complexity_explanation)
        )
        
        return {
            "summary": (
                f"This {language} code contains {ast_analysis.get('num_functions', 0)} "
                f"functions and {ast_analysis.get('num_classes', 0)} classes."
            ),
            "complexity_analysis": {
                "time_complexity": {
                    "display": complexity_info["display"],
                    "html": complexity_info["html"]
                },
                "space_complexity": space_complexity,
                "explanation": complexity_explanation,
                "details": {
                    "time": (f"The time complexity is {complexity_info['display']}, which indicates "
                            "how the execution time grows with input size."),
                    "space": (f"The space complexity is {space_complexity}, which "
                             "indicates the memory usage pattern.")
                }
            },
            "structure_analysis": ast_analysis,
            "suggested_improvements": suggestions,
            "learning_resources": resources
        }
    except Exception as e:
        print(f"Analysis error: {str(e)}")
        return {
            "error": "Analysis failed",
            "complexity_analysis": {
                "time_complexity": {
                    "display": "O(1)",
                    "html": "O(1)"
                },
                "space_complexity": "O(1)",
                "explanation": "Analysis failed"
            }
        }

# Flask routes
@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        "status": "healthy",
        "timestamp": time.time(),
        "model_loaded": hasattr(app, 'complexity_analyzer')
    })

@app.route('/analyze', methods=['POST'])
def analyze_code():
    """Analyze code endpoint."""
    try:
        data = request.json
        code = data.get('code', '')
        language = data.get('language', '')
        
        if not code or not language:
            return jsonify({
                "error": "Missing code or language",
                "time_complexity": {
                    "display": "O(1)",
                    "html": "O(1)"
                },
                "space_complexity": "O(1)"
            }), 400
            
        analysis = generate_detailed_analysis(code, language)
        
        return jsonify({
            "time_complexity": analysis.get("complexity_analysis", {}).get("time_complexity", {
                "display": "O(1)",
                "html": "O(1)"
            }),
            "space_complexity": analysis.get("complexity_analysis", {}).get("space_complexity", "O(1)"),
            "complexity_explanation": analysis.get("complexity_analysis", {}).get("explanation", ""),
            "analysis": analysis
        })
    
    except Exception as e:
        return jsonify({
            "error": str(e),
            "time_complexity": {
                "display": "O(1)",
                "html": "O(1)"
            },
            "space_complexity": "O(1)",
            "complexity_explanation": "Analysis failed"
        }), 500

@app.route('/execute', methods=['POST'])
def execute_code():
    """Execute code endpoint using Judge0 API."""
    try:
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "Invalid or missing JSON"}), 400
        
        code = data.get('code', '')
        language = data.get('language', '')
        if not code or not language:
            return jsonify({"error": "Missing code or language"}), 400
        
        language_id = LANGUAGE_MAP.get(language)
        if not language_id:
            return jsonify({"error": f"Unsupported language: {language}"}), 400
        
        # Generate code analysis
        analysis = generate_detailed_analysis(code, language)
        
        # Prepare submission data
        submission_data = {
            "source_code": code,
            "language_id": language_id,
            "stdin": data.get('stdin', '')
        }
        
        # Submit code to Judge0 API
        try:
            response = requests.post(
                JUDGE0_API_URL,
                json=submission_data,
                headers=HEADERS,
                timeout=10
            )
            response.raise_for_status()
        except requests.RequestException as e:
            return jsonify({
                "error": f"Code execution submission error: {str(e)}",
                "analysis": analysis
            }), 500
        
        token = response.json().get("token")
        if not token:
            return jsonify({
                "error": "Failed to retrieve submission token",
                "analysis": analysis
            }), 500
        
        # Poll for results
        max_attempts = 10
        for attempt in range(max_attempts):
            time.sleep(1)
            try:
                result = requests.get(
                    f"{JUDGE0_API_URL}/{token}",
                    headers=HEADERS,
                    timeout=10
                ).json()
            except requests.RequestException as e:
                return jsonify({
                    "error": f"Error fetching execution result: {str(e)}",
                    "analysis": analysis
                }), 500
            
            status_id = result.get("status", {}).get("id")
            if status_id not in [1, 2]:  # Not queued or processing
                output_parts = []
                if result.get("stdout"):
                    output_parts.append(result["stdout"])
                if result.get("stderr"):
                    output_parts.append(result["stderr"])
                if result.get("compile_output"):
                    output_parts.append(f"Compilation Output:\n{result['compile_output']}")
                    
                return jsonify({
                    "output": "\n\n".join(output_parts) if output_parts else "No output",
                    "analysis": analysis
                })
        
        return jsonify({
            "error": "Execution timeout",
            "analysis": analysis
        }), 408
    
    except Exception as e:
        return jsonify({"error": f"Internal server error: {str(e)}",
            "analysis": generate_detailed_analysis(code, language) if code and language else None
        }), 500

# Initialize analyzers
complexity_analyzer = None
code_analyzer = None

def initialize_app():
    """Initialize the application by loading the model and creating analyzers."""
    global complexity_analyzer, code_analyzer
    try:
        print("Loading model and initializing analyzers...")
        complexity_analyzer = ComplexityAnalyzer(MODEL_DIR)
        code_analyzer = CodeAnalyzer()
        print("Initialization completed successfully!")
    except Exception as e:
        print(f"Error initializing application: {str(e)}")
        raise

# Initialize the application when running directly
if __name__ == '__main__':
    initialize_app()
    app.run(debug=True, host='0.0.0.0', port=5000)