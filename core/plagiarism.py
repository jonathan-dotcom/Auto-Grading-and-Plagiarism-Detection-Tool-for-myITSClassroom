import os
import ast
import tokenize
import re
import io
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class PlagiarismDetector:
    """Base class for plagiarism detection."""

    def __init__(self, threshold: float = 0.8):
        """Initialize the plagiarism detector.

        Args:
            threshold: Similarity threshold to flag as plagiarism
        """
        self.threshold = threshold

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between submissions.

        Args:
            submissions: List of submission dictionaries.
                         Each dictionary is expected to have 'student_id', 'student_name',
                         'file_name', and 'content' (the actual text or code content).

        Returns:
            List of plagiarism results
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def generate_report(self, results: List[Dict[str, Any]], total_submissions_compared: int) -> Dict[str, Any]:
        """Generate a plagiarism report.

        Args:
            results: Plagiarism detection results (list of compared pairs).
            total_submissions_compared: The number of unique submissions that were part of the comparison.

        Returns:
            Report data
        """
        flagged_pairs = [r for r in results if r.get('flagged', False)]
        flagged_student_ids = set()
        for pair in flagged_pairs:
            flagged_student_ids.add(pair['student1_id'])
            flagged_student_ids.add(pair['student2_id'])

        summary = {
            'total_submissions': total_submissions_compared,
            'total_comparisons': len(results),
            'flagged_pairs': len(flagged_pairs),
            'flagged_student_ids': list(flagged_student_ids),
            'threshold': self.threshold,
            'results': results
        }
        return summary


class TextPlagiarismDetector(PlagiarismDetector):
    """Detect plagiarism in text submissions using NLP techniques."""

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between text submissions using TF-IDF and cosine similarity.

        Args:
            submissions: List of submission dictionaries, each with 'student_id',
                         'student_name', 'file_name', and 'content'.

        Returns:
            List of plagiarism detection results (compared pairs).
        """
        texts = []
        metadata = []

        for submission in submissions:
            content = submission.get('content')
            if content is None:
                print(
                    f"Warning: Content missing for submission from student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}")
                continue

            texts.append(content)
            metadata.append({
                'student_id': submission['student_id'],
                'student_name': submission['student_name'],
                'file_name': submission['file_name']
            })

        if len(texts) < 2:
            return []

        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=2 if len(texts) > 10 else 1
        )

        try:
            if not any(texts):
                print("Warning: All text submissions are empty or None.")
                return []

            tfidf_matrix = vectorizer.fit_transform(texts)

            if tfidf_matrix.shape[0] == 0 or tfidf_matrix.shape[1] == 0:
                print("Warning: TF-IDF matrix is empty. Could not vectorize texts.")
                return []

            similarity_matrix = cosine_similarity(tfidf_matrix)
            results = []
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    similarity = similarity_matrix[i][j]
                    results.append({
                        'student1_id': metadata[i]['student_id'],
                        'student1_name': metadata[i]['student_name'],
                        'student1_file': metadata[i]['file_name'],
                        'student2_id': metadata[j]['student_id'],
                        'student2_name': metadata[j]['student_name'],
                        'student2_file': metadata[j]['file_name'],
                        'similarity': float(similarity),
                        'flagged': similarity >= self.threshold
                    })
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results
        except ValueError as ve:
            print(f"ValueError in text plagiarism detection (TF-IDF): {ve}. This might happen if vocabulary is empty.")
            return []
        except Exception as e:
            print(f"Error in text plagiarism detection: {e}")
            return []


class CodePlagiarismDetector(PlagiarismDetector):
    """Detect plagiarism in code submissions using AST and token analysis."""

    def __init__(self, threshold: float = 0.8, language: str = 'python'):
        super().__init__(threshold)
        self.language = language.lower()  # Ensure lowercase for consistency

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.language == 'python':
            return self._detect_python_plagiarism(submissions)
        elif self.language == 'c':
            return self._detect_c_plagiarism(submissions)
        elif self.language == 'cpp' or self.language == 'c++':
            return self._detect_cpp_plagiarism(submissions)
        else:
            print(f"Warning: Plagiarism detection for {self.language} not implemented yet. Returning empty results.")
            return []

    def _detect_python_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                print(
                    f"Warning: Code content missing for submission from student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}")
                continue

            try:
                processed = self._process_python_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'tokens': processed['tokens'],
                        'ast_nodes': processed['ast_nodes']
                    })
            except Exception as e:
                print(
                    f"Error processing Python code for student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                token_sim = self._calculate_token_similarity(sub1['tokens'], sub2['tokens'])
                ast_sim = self._calculate_ast_similarity(sub1['ast_nodes'], sub2['ast_nodes'])

                token_weight = 0.6
                ast_weight = 0.4
                combined_sim = token_weight * token_sim + ast_weight * ast_sim

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'token_similarity': float(token_sim),
                    'ast_similarity': float(ast_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _detect_c_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between C code submissions.

        Args:
            submissions: List of submission dictionaries.

        Returns:
            List of plagiarism detection results.
        """
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                print(
                    f"Warning: Code content missing for submission from student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}")
                continue

            try:
                processed = self._process_c_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'tokens': processed['tokens'],
                        'structures': processed['structures']
                    })
            except Exception as e:
                print(
                    f"Error processing C code for student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                token_sim = self._calculate_token_similarity(sub1['tokens'], sub2['tokens'])
                struct_sim = self._calculate_structure_similarity(sub1['structures'], sub2['structures'])

                token_weight = 0.6
                struct_weight = 0.4
                combined_sim = token_weight * token_sim + struct_weight * struct_sim

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'token_similarity': float(token_sim),
                    'struct_similarity': float(struct_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _detect_cpp_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between C++ code submissions.

        Args:
            submissions: List of submission dictionaries.

        Returns:
            List of plagiarism detection results.
        """
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                print(
                    f"Warning: Code content missing for submission from student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}")
                continue

            try:
                processed = self._process_cpp_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'tokens': processed['tokens'],
                        'structures': processed['structures']
                    })
            except Exception as e:
                print(
                    f"Error processing C++ code for student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                token_sim = self._calculate_token_similarity(sub1['tokens'], sub2['tokens'])
                struct_sim = self._calculate_structure_similarity(sub1['structures'], sub2['structures'])

                token_weight = 0.6
                struct_weight = 0.4
                combined_sim = token_weight * token_sim + struct_weight * struct_sim

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'token_similarity': float(token_sim),
                    'struct_similarity': float(struct_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _process_python_code(self, code: str) -> Dict[str, Any] | None:
        """Process Python code to extract tokens and AST nodes."""
        try:
            tokens = []
            token_generator = tokenize.generate_tokens(io.StringIO(code).readline)
            for token in token_generator:
                if token.type not in (
                        tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL,
                        tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER
                ):
                    tokens.append((tokenize.tok_name[token.type], token.string))

            if not code.strip():
                return {'tokens': [], 'ast_nodes': []}

            tree = ast.parse(code)
            ast_nodes = []
            for node in ast.walk(tree):
                node_info = {'type': type(node).__name__}
                if isinstance(node, ast.Name):
                    node_info['id'] = node.id
                elif isinstance(node, ast.Constant):
                    node_info['value'] = str(node.value)
                elif isinstance(node, (ast.Num, ast.Str, ast.Bytes, ast.NameConstant)):
                    if isinstance(node, ast.Num):
                        node_info['value'] = str(node.n)
                    elif isinstance(node, ast.Str):
                        node_info['value'] = node.s
                    elif isinstance(node, ast.Bytes):
                        node_info['value'] = str(node.s)
                    elif isinstance(node, ast.NameConstant):
                        node_info['value'] = str(node.value)
                elif isinstance(node, ast.Attribute):
                    node_info['attr'] = node.attr
                elif isinstance(node, ast.FunctionDef):
                    node_info['name'] = node.name
                    node_info['args_count'] = len(node.args.args)
                elif isinstance(node, ast.ClassDef):
                    node_info['name'] = node.name
                    node_info['bases_count'] = len(node.bases)

                ast_nodes.append(node_info)
            return {'tokens': tokens, 'ast_nodes': ast_nodes}
        except SyntaxError:
            print(f"Syntax error in Python code, cannot parse AST or tokens.")
            return None
        except tokenize.TokenError as e:
            print(f"Tokenization error in Python code: {e}")
            return None
        except Exception as e:
            print(f"Unexpected error processing Python code: {e}")
            return None

    def _process_c_code(self, code: str) -> Dict[str, Any]:
        """Process C code to extract tokens and structure information.

        Args:
            code: C code as string.

        Returns:
            Dictionary with tokens and structures.
        """
        # Remove comments
        code = self._remove_c_comments(code)

        # Extract tokens
        tokens = self._tokenize_c_code(code)

        # Extract structures (functions, loops, conditionals)
        structures = self._extract_c_structures(code)

        return {'tokens': tokens, 'structures': structures}

    def _process_cpp_code(self, code: str) -> Dict[str, Any]:
        """Process C++ code to extract tokens and structure information.

        Args:
            code: C++ code as string.

        Returns:
            Dictionary with tokens and structures.
        """
        # For C++, we can reuse most of the C processing with some extensions
        # Remove comments
        code = self._remove_c_comments(code)

        # Extract tokens with C++ specific ones
        tokens = self._tokenize_cpp_code(code)

        # Extract structures (classes, functions, loops, conditionals)
        structures = self._extract_cpp_structures(code)

        return {'tokens': tokens, 'structures': structures}

    def _remove_c_comments(self, code: str) -> str:
        """Remove C-style comments from code.

        Args:
            code: Original code with comments.

        Returns:
            Code with comments removed.
        """
        # Remove multi-line comments (/* ... */)
        code = re.sub(r'/\*.*?\*/', ' ', code, flags=re.DOTALL)

        # Remove single-line comments (// ...)
        code = re.sub(r'//.*?$', ' ', code, flags=re.MULTILINE)

        return code

    def _tokenize_c_code(self, code: str) -> List[Tuple[str, str]]:
        """Tokenize C code into meaningful tokens.

        Args:
            code: C code as string.

        Returns:
            List of token types and values.
        """
        # Define token patterns
        token_patterns = [
            ('KEYWORD',
             r'\b(auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|int|long|register|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while)\b'),
            ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('NUMBER', r'\b\d+(\.\d+)?(e[+-]?\d+)?\b'),
            ('STRING', r'"[^"\\]*(\\.[^"\\]*)*"'),
            ('CHAR', r"'[^'\\]*(\\.[^'\\]*)*'"),
            ('OPERATOR', r'[+\-*/=%&|^~!<>?:]+'),
            ('PUNCTUATION', r'[(){}\[\],;.]'),
            ('WHITESPACE', r'\s+')
        ]

        # Combine patterns
        combined_pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_patterns)

        # Tokenize
        tokens = []
        for match in re.finditer(combined_pattern, code):
            token_type = match.lastgroup
            token_value = match.group()

            # Skip whitespace
            if token_type != 'WHITESPACE':
                tokens.append((token_type, token_value))

        return tokens

    def _tokenize_cpp_code(self, code: str) -> List[Tuple[str, str]]:
        """Tokenize C++ code into meaningful tokens.

        Args:
            code: C++ code as string.

        Returns:
            List of token types and values.
        """
        # Define token patterns with C++ specific keywords added
        token_patterns = [
            ('KEYWORD',
             r'\b(auto|break|case|char|const|continue|default|do|double|else|enum|extern|float|for|goto|if|int|long|register|return|short|signed|sizeof|static|struct|switch|typedef|union|unsigned|void|volatile|while|class|namespace|template|try|catch|throw|new|delete|this|private|protected|public|friend|virtual|inline|bool|true|false|operator|using|explicit|export|mutable|typename)\b'),
            ('IDENTIFIER', r'[a-zA-Z_][a-zA-Z0-9_]*'),
            ('NUMBER', r'\b\d+(\.\d+)?(e[+-]?\d+)?\b'),
            ('STRING', r'"[^"\\]*(\\.[^"\\]*)*"'),
            ('CHAR', r"'[^'\\]*(\\.[^'\\]*)*'"),
            ('OPERATOR', r'[+\-*/=%&|^~!<>?:]+|::|\->|<<|>>|&&|\|\||\+\+|\-\-'),
            ('PUNCTUATION', r'[(){}\[\],;.]'),
            ('WHITESPACE', r'\s+')
        ]

        # Combine patterns
        combined_pattern = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in token_patterns)

        # Tokenize
        tokens = []
        for match in re.finditer(combined_pattern, code):
            token_type = match.lastgroup
            token_value = match.group()

            # Skip whitespace
            if token_type != 'WHITESPACE':
                tokens.append((token_type, token_value))

        return tokens

    def _extract_c_structures(self, code: str) -> List[Dict[str, Any]]:
        """Extract structural elements from C code.

        Args:
            code: C code as string.

        Returns:
            List of structural elements.
        """
        structures = []

        # Extract functions
        function_pattern = r'\b(\w+)\s+(\w+)\s*\((.*?)\)\s*\{([^}]*)\}'
        for match in re.finditer(function_pattern, code, re.DOTALL):
            return_type, name, params, body = match.groups()

            # Count parameters
            param_count = len(params.split(',')) if params.strip() else 0

            # Analyze function complexity (count loops, conditionals)
            loops = len(re.findall(r'\b(for|while|do)\b', body))
            conditionals = len(re.findall(r'\b(if|else|switch|case)\b', body))

            structures.append({
                'type': 'function',
                'name': name,
                'return_type': return_type,
                'param_count': param_count,
                'body_length': len(body),
                'loops': loops,
                'conditionals': conditionals
            })

        # Extract loops (for, while, do-while)
        loop_patterns = [
            (r'for\s*\((.*?);(.*?);(.*?)\)\s*\{([^}]*)\}', 'for'),
            (r'while\s*\((.*?)\)\s*\{([^}]*)\}', 'while'),
            (r'do\s*\{([^}]*)\}\s*while\s*\((.*?)\);', 'do-while')
        ]

        for pattern, loop_type in loop_patterns:
            for match in re.finditer(pattern, code, re.DOTALL):
                structures.append({
                    'type': 'loop',
                    'loop_type': loop_type,
                    'body_length': len(match.group())
                })

        # Extract conditionals (if, if-else, switch)
        if_pattern = r'if\s*\((.*?)\)\s*\{([^}]*)\}'
        for match in re.finditer(if_pattern, code, re.DOTALL):
            condition, body = match.groups()
            structures.append({
                'type': 'conditional',
                'condition_type': 'if',
                'body_length': len(body)
            })

        if_else_pattern = r'if\s*\((.*?)\)\s*\{([^}]*)\}\s*else\s*\{([^}]*)\}'
        for match in re.finditer(if_else_pattern, code, re.DOTALL):
            condition, if_body, else_body = match.groups()
            structures.append({
                'type': 'conditional',
                'condition_type': 'if-else',
                'if_body_length': len(if_body),
                'else_body_length': len(else_body)
            })

        switch_pattern = r'switch\s*\((.*?)\)\s*\{([^}]*)\}'
        for match in re.finditer(switch_pattern, code, re.DOTALL):
            expression, body = match.groups()
            # Count case statements
            cases = len(re.findall(r'\bcase\b', body))
            structures.append({
                'type': 'conditional',
                'condition_type': 'switch',
                'cases': cases,
                'body_length': len(body)
            })

        return structures

    def _extract_cpp_structures(self, code: str) -> List[Dict[str, Any]]:
        """Extract structural elements from C++ code.

        Args:
            code: C++ code as string.

        Returns:
            List of structural elements.
        """
        # Start with C structures
        structures = self._extract_c_structures(code)

        # Extract C++ specific structures

        # Classes
        class_pattern = r'class\s+(\w+)(?:\s*:\s*(?:public|protected|private)\s+(\w+))?\s*\{([^}]*)\}'
        for match in re.finditer(class_pattern, code, re.DOTALL):
            name, parent, body = match.groups()

            # Count methods and fields
            methods = len(re.findall(r'\b\w+\s+\w+\s*\([^)]*\)\s*(?:const)?\s*(?:=\s*0)?\s*[;{]', body))
            fields = len(re.findall(r'\b\w+\s+\w+\s*;', body))

            structures.append({
                'type': 'class',
                'name': name,
                'parent': parent if parent else 'None',
                'methods': methods,
                'fields': fields,
                'body_length': len(body)
            })

        # Namespaces
        namespace_pattern = r'namespace\s+(\w+)\s*\{([^}]*)\}'
        for match in re.finditer(namespace_pattern, code, re.DOTALL):
            name, body = match.groups()
            structures.append({
                'type': 'namespace',
                'name': name,
                'body_length': len(body)
            })

        # Templates
        template_pattern = r'template\s*<([^>]*)>\s*(?:class|struct)\s+(\w+)(?:\s*:\s*(?:public|protected|private)\s+(\w+))?\s*\{([^}]*)\}'
        for match in re.finditer(template_pattern, code, re.DOTALL):
            params, name, parent, body = match.groups()

            # Count template parameters
            param_count = len(params.split(',')) if params.strip() else 0

            structures.append({
                'type': 'template_class',
                'name': name,
                'parent': parent if parent else 'None',
                'param_count': param_count,
                'body_length': len(body)
            })

        return structures

    def _calculate_token_similarity(self, tokens1: List[Tuple[str, str]], tokens2: List[Tuple[str, str]]) -> float:

        str_tokens1 = ["{}:{}".format(token_type, token_val) for token_type, token_val in tokens1]
        str_tokens2 = ["{}:{}".format(token_type, token_val) for token_type, token_val in tokens2]

        if not str_tokens1 and not str_tokens2: return 1.0
        if not str_tokens1 or not str_tokens2: return 0.0

        corpus = [' '.join(str_tokens1), ' '.join(str_tokens2)]

        try:
            vectorizer = TfidfVectorizer(min_df=1)
            tfidf_matrix = vectorizer.fit_transform(corpus)
            if tfidf_matrix.shape[1] == 0:
                return 0.0
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except ValueError:
            set1 = set(str_tokens1)
            set2 = set(str_tokens2)
            if not set1 and not set2: return 1.0
            if not set1 or not set2: return 0.0

            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0
        except Exception as e:
            print(f"Error in _calculate_token_similarity: {e}")
            return 0.0

    def _calculate_ast_similarity(self, nodes1: List[Dict[str, Any]], nodes2: List[Dict[str, Any]]) -> float:

        def get_node_type_counts(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
            counts = {}
            for node in nodes:
                node_type = node.get('type', 'UnknownNode')
                counts[node_type] = counts.get(node_type, 0) + 1
            return counts

        counts1 = get_node_type_counts(nodes1)
        counts2 = get_node_type_counts(nodes2)

        if not counts1 and not counts2: return 1.0
        if not counts1 or not counts2: return 0.0

        all_node_types = set(counts1.keys()).union(set(counts2.keys()))
        if not all_node_types: return 1.0

        vec1 = np.array([counts1.get(nt, 0) for nt in all_node_types])
        vec2 = np.array([counts2.get(nt, 0) for nt in all_node_types])

        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 1.0 if norm_vec1 == norm_vec2 else 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    def _calculate_structure_similarity(self, structures1: List[Dict[str, Any]],
                                        structures2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between code structures.

        Args:
            structures1: First list of structure dictionaries.
            structures2: Second list of structure dictionaries.

        Returns:
            Similarity score between 0 and 1.
        """
        if not structures1 and not structures2:
            return 1.0  # Both are empty, perfect match

        if not structures1 or not structures2:
            return 0.0  # One is empty, no match

        # Count types of structures
        def count_structure_types(structures):
            counts = {}
            for struct in structures:
                struct_type = struct['type']
                counts[struct_type] = counts.get(struct_type, 0) + 1
            return counts

        type_counts1 = count_structure_types(structures1)
        type_counts2 = count_structure_types(structures2)

        # Get all unique structure types
        all_types = set(type_counts1.keys()) | set(type_counts2.keys())

        # Create vectors from counts
        vec1 = np.array([type_counts1.get(t, 0) for t in all_types])
        vec2 = np.array([type_counts2.get(t, 0) for t in all_types])

        # Calculate cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        type_similarity = dot_product / (norm1 * norm2)

        # Function similarity - compare function names and complexity
        function_similarity = 0.0
        functions1 = [s for s in structures1 if s['type'] == 'function']
        functions2 = [s for s in structures2 if s['type'] == 'function']

        if functions1 and functions2:
            # Create a set of function names in each submission
            names1 = set(f['name'] for f in functions1)
            names2 = set(f['name'] for f in functions2)

            # Check name overlap
            common_names = names1.intersection(names2)
            name_similarity = len(common_names) / max(len(names1), len(names2)) if max(len(names1),
                                                                                       len(names2)) > 0 else 0

            # Compare function complexity for common functions
            complexity_similarity = 0.0
            if common_names:
                complexities = []
                for name in common_names:
                    f1 = next(f for f in functions1 if f['name'] == name)
                    f2 = next(f for f in functions2 if f['name'] == name)

                    # Compare parameters, loops, conditionals
                    param_sim = 1.0 if f1['param_count'] == f2['param_count'] else 0.0
                    loops_sim = 1.0 if f1['loops'] == f2['loops'] else 0.0
                    cond_sim = 1.0 if f1['conditionals'] == f2['conditionals'] else 0.0

                    # Weighted average
                    func_sim = (0.3 * param_sim + 0.35 * loops_sim + 0.35 * cond_sim)
                    complexities.append(func_sim)

                complexity_similarity = sum(complexities) / len(complexities) if complexities else 0

            function_similarity = 0.6 * name_similarity + 0.4 * complexity_similarity

        # Combine type and function similarity
        combined_similarity = 0.5 * type_similarity + 0.5 * function_similarity

        return combined_similarity


class PlagiarismManager:
    """Manage plagiarism detection for different types of submissions."""

    def __init__(self):
        self.detectors: Dict[str, PlagiarismDetector] = {}

    def register_detector(self, submission_type: str, detector: PlagiarismDetector):
        self.detectors[submission_type] = detector

    def detect_plagiarism(self, submission_type: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Detect plagiarism for a given submission type.

        Args:
            submission_type: The type of submission (e.g., 'text', 'code').
            submissions: A list of submission dictionaries. Each dictionary should contain
                         'student_id', 'student_name', 'file_name', and 'content'.

        Returns:
            A plagiarism report dictionary.
        """
        if submission_type not in self.detectors:
            # Consider logging an error or returning a specific error structure
            print(f"Error: No detector registered for submission type '{submission_type}'")
            return {  # Return a default empty report structure
                'total_submissions': len(submissions),
                'total_comparisons': 0,
                'flagged_pairs': 0,
                'flagged_student_ids': [],
                'threshold': 0.0,
                'results': [],
                'error': f"No detector registered for {submission_type}"
            }

        detector = self.detectors[submission_type]
        comparison_results = detector.detect(submissions)

        report = detector.generate_report(comparison_results, total_submissions_compared=len(submissions))

        return report