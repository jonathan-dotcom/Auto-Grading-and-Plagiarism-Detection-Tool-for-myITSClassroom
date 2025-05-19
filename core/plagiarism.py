import os
import ast
import tokenize
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
                print(f"Warning: Content missing for submission from student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}")
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
        self.language = language

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if self.language == 'python':
            return self._detect_python_plagiarism(submissions)
        else:
            print(f"Warning: Plagiarism detection for {self.language} not implemented yet. Returning empty results.")
            return []


    def _detect_python_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                print(f"Warning: Code content missing for submission from student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}")
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
                print(f"Error processing Python code for student {submission.get('student_id', 'Unknown')}, file {submission.get('file_name', 'Unknown')}: {e}")

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
                    if isinstance(node, ast.Num): node_info['value'] = str(node.n)
                    elif isinstance(node, ast.Str): node_info['value'] = node.s
                    elif isinstance(node, ast.Bytes): node_info['value'] = str(node.s)
                    elif isinstance(node, ast.NameConstant): node_info['value'] = str(node.value)
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
            return { # Return a default empty report structure
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
