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
            submissions: List of submission dictionaries

        Returns:
            List of plagiarism results
        """
        raise NotImplementedError("Subclasses must implement detect method")

    def generate_report(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a plagiarism report.

        Args:
            results: Plagiarism detection results

        Returns:
            Report data
        """
        # Count submissions with high similarity
        flagged_pairs = [r for r in results if r['similarity'] >= self.threshold]
        flagged_students = set()
        for pair in flagged_pairs:
            flagged_students.add(pair['student1_id'])
            flagged_students.add(pair['student2_id'])

        summary = {
            'total_submissions': len(results),
            'flagged_pairs': len(flagged_pairs),
            'flagged_students': len(flagged_students),
            'threshold': self.threshold,
            'results': results
        }

        return summary


class TextPlagiarismDetector(PlagiarismDetector):
    """Detect plagiarism in text submissions using NLP techniques."""

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between text submissions using TF-IDF and cosine similarity.

        Args:
            submissions: List of submission dictionaries with content

        Returns:
            List of plagiarism detection results
        """
        # Extract text content and metadata
        texts = []
        metadata = []

        for submission in submissions:
            file_path = submission['current_file']['path']
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    texts.append(content)
                    metadata.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['current_file']['name']
                    })
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")

        # Check if we have submissions to compare
        if len(texts) < 2:
            return []

        # Calculate TF-IDF vectors
        vectorizer = TfidfVectorizer(
            lowercase=True,
            stop_words='english',
            ngram_range=(1, 3),  # Use unigrams, bigrams, and trigrams
            max_df=0.85,
            min_df=0.1
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(texts)

            # Calculate pairwise cosine similarity
            similarity_matrix = cosine_similarity(tfidf_matrix)

            # Generate results
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

            # Sort by similarity (descending)
            results.sort(key=lambda x: x['similarity'], reverse=True)
            return results

        except Exception as e:
            print(f"Error in text plagiarism detection: {e}")
            return []


class CodePlagiarismDetector(PlagiarismDetector):
    """Detect plagiarism in code submissions using AST and token analysis."""

    def __init__(self, threshold: float = 0.8, language: str = 'python'):
        """Initialize the code plagiarism detector.

        Args:
            threshold: Similarity threshold to flag as plagiarism
            language: Programming language of the submissions
        """
        super().__init__(threshold)
        self.language = language

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between code submissions.

        Args:
            submissions: List of submission dictionaries with code content

        Returns:
            List of plagiarism detection results
        """
        # Filter submissions by language
        if self.language == 'python':
            return self._detect_python_plagiarism(submissions)
        else:
            # For other languages, implement specific detection logic
            raise NotImplementedError(f"Plagiarism detection for {self.language} not implemented yet")

    def _detect_python_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between Python code submissions using AST and tokenization.

        Args:
            submissions: List of submission dictionaries with Python code

        Returns:
            List of plagiarism detection results
        """
        # Extract code content and metadata
        processed_submissions = []

        for submission in submissions:
            file_path = submission['current_file']['path']
            try:
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()

                    # Process the code
                    processed = self._process_python_code(content)

                    if processed:
                        processed_submissions.append({
                            'student_id': submission['student_id'],
                            'student_name': submission['student_name'],
                            'file_name': submission['current_file']['name'],
                            'tokens': processed['tokens'],
                            'ast_nodes': processed['ast_nodes']
                        })
            except Exception as e:
                print(f"Error processing Python file {file_path}: {e}")

        # Check if we have submissions to compare
        if len(processed_submissions) < 2:
            return []

        # Compare submissions
        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                # Calculate token similarity
                token_sim = self._calculate_token_similarity(sub1['tokens'], sub2['tokens'])

                # Calculate AST similarity
                ast_sim = self._calculate_ast_similarity(sub1['ast_nodes'], sub2['ast_nodes'])

                # Combine similarities (weighted average)
                combined_sim = 0.6 * token_sim + 0.4 * ast_sim

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

        # Sort by similarity (descending)
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _process_python_code(self, code: str) -> Dict[str, Any]:
        """Process Python code to extract tokens and AST nodes.

        Args:
            code: Python code as string

        Returns:
            Dictionary with tokens and AST nodes
        """
        try:
            # Extract tokens
            tokens = []
            token_generator = tokenize.generate_tokens(io.StringIO(code).readline)
            for token in token_generator:
                if token.type not in (
                tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL, tokenize.INDENT, tokenize.DEDENT):
                    tokens.append((tokenize.tok_name[token.type], token.string))

            # Parse AST
            tree = ast.parse(code)
            ast_nodes = []
            for node in ast.walk(tree):
                # Get node type and relevant attributes
                node_info = {
                    'type': type(node).__name__
                }

                # Add relevant attributes based on node type
                if isinstance(node, ast.Name):
                    node_info['id'] = node.id
                elif isinstance(node, ast.FunctionDef):
                    node_info['name'] = node.name
                    node_info['args'] = len(node.args.args)
                elif isinstance(node, ast.ClassDef):
                    node_info['name'] = node.name
                    node_info['bases'] = len(node.bases)

                ast_nodes.append(node_info)

            return {
                'tokens': tokens,
                'ast_nodes': ast_nodes
            }
        except Exception as e:
            print(f"Error processing Python code: {e}")
            return None

    def _calculate_token_similarity(self, tokens1: List[Tuple[str, str]], tokens2: List[Tuple[str, str]]) -> float:
        """Calculate similarity between two token sequences.

        Args:
            tokens1: First token sequence
            tokens2: Second token sequence

        Returns:
            Similarity score between 0 and 1
        """
        # Convert tokens to strings for vectorization
        token_strings1 = [f"{t[0]}:{t[1]}" for t in tokens1]
        token_strings2 = [f"{t[0]}:{t[1]}" for t in tokens2]

        # Use TF-IDF and cosine similarity
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 3))
        try:
            tfidf_matrix = vectorizer.fit_transform([' '.join(token_strings1), ' '.join(token_strings2)])
            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            # Fallback to Jaccard similarity if vectorization fails
            set1 = set(token_strings1)
            set2 = set(token_strings2)

            if not set1 or not set2:
                return 0.0

            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))

            return intersection / union if union > 0 else 0.0

    def _calculate_ast_similarity(self, nodes1: List[Dict[str, Any]], nodes2: List[Dict[str, Any]]) -> float:
        """Calculate similarity between two AST node sequences.

        Args:
            nodes1: First AST node sequence
            nodes2: Second AST node sequence

        Returns:
            Similarity score between 0 and 1
        """

        # Count node types
        def count_node_types(nodes):
            counts = {}
            for node in nodes:
                node_type = node['type']
                if node_type in counts:
                    counts[node_type] += 1
                else:
                    counts[node_type] = 1
            return counts

        counts1 = count_node_types(nodes1)
        counts2 = count_node_types(nodes2)

        # Get all node types
        all_types = set(counts1.keys()).union(counts2.keys())

        # Calculate cosine similarity between count vectors
        vec1 = np.array([counts1.get(t, 0) for t in all_types])
        vec2 = np.array([counts2.get(t, 0) for t in all_types])

        # Normalize vectors
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)


class PlagiarismManager:
    """Manage plagiarism detection for different types of submissions."""

    def __init__(self):
        """Initialize the plagiarism manager."""
        self.detectors = {}

    def register_detector(self, submission_type: str, detector: PlagiarismDetector):
        """Register a detector for a specific submission type.

        Args:
            submission_type: Type of submission (e.g., 'text', 'code')
            detector: Detector instance
        """
        self.detectors[submission_type] = detector

    def detect_plagiarism(self, submission_type: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect plagiarism between submissions.

        Args:
            submission_type: Type of submission
            submissions: List of submissions

        Returns:
            Plagiarism report
        """
        if submission_type not in self.detectors:
            raise ValueError(f"No detector registered for {submission_type}")

        results = self.detectors[submission_type].detect(submissions)
        report = self.detectors[submission_type].generate_report(results)

        return report