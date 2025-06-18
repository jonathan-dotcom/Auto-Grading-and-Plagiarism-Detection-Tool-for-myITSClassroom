import os
import ast
import tokenize
import io
import re
import hashlib
import time
import logging
import copy
from typing import List, Dict, Any, Tuple, Set, Optional
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('plagiarism_detection')


class PlagiarismDetector:
    """Base class for plagiarism detection."""

    def __init__(self, threshold: float = 0.8, weights: Optional[Dict[str, float]] = None):
        """Initialize the plagiarism detector.

        Args:
            threshold: Similarity threshold to flag as plagiarism
            weights: Custom weights for combining similarity metrics
        """
        self.threshold = threshold
        self.weights = weights or {}

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
            'weights': self.weights,
            'results': results
        }
        return summary


class TextPlagiarismDetector(PlagiarismDetector):
    """Detect plagiarism in text submissions using advanced NLP techniques."""

    def __init__(self, threshold: float = 0.8, weights: Optional[Dict[str, float]] = None):
        """Initialize with configurable weights."""
        default_weights = {
            'tfidf': 0.4,
            'fingerprint': 0.3,
            'ngram': 0.3
        }
        if weights:
            default_weights.update(weights)
        super().__init__(threshold, default_weights)
        self.fingerprint_size = 10

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between text submissions using multiple techniques."""
        start_time = time.time()
        logger.info(f"Starting text plagiarism detection with {len(submissions)} submissions")
        logger.debug(f"Threshold: {self.threshold}, Weights: {self.weights}")

        texts = []
        metadata = []

        for i, submission in enumerate(submissions):
            logger.debug(f"Processing submission {i + 1}: {submission.get('student_name', 'Unknown')}")
            content = submission.get('content')
            if content is None:
                logger.warning(f"Content missing for submission from student {submission.get('student_id', 'Unknown')}")
                continue

            cleaned_content = self._clean_text_content(content)
            logger.debug(f"Original content length: {len(content)}, Cleaned length: {len(cleaned_content)}")

            if not cleaned_content or len(cleaned_content.strip()) < 50:
                logger.warning(
                    f"Insufficient content for student {submission.get('student_name', 'Unknown')} - only {len(cleaned_content)} chars")
                continue

            texts.append(cleaned_content)
            metadata.append({
                'student_id': submission['student_id'],
                'student_name': submission['student_name'],
                'file_name': submission['file_name'],
                'content': content
            })

        logger.info(f"Processing {len(texts)} text submissions for plagiarism detection after filtering")

        if len(texts) < 2:
            logger.warning("Not enough submissions for comparison")
            return []

        try:
            results = []
            total_comparisons = (len(texts) * (len(texts) - 1)) // 2
            logger.info(f"Will perform {total_comparisons} pairwise comparisons")

            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    student1 = metadata[i]['student_name']
                    student2 = metadata[j]['student_name']

                    logger.debug(f"Comparing: {student1} vs {student2}")

                    # Multiple similarity measures
                    tfidf_sim = self._calculate_tfidf_similarity(texts[i], texts[j])
                    fingerprint_sim = self._calculate_fingerprint_similarity(texts[i], texts[j])
                    ngram_sim = self._calculate_ngram_similarity(texts[i], texts[j])

                    # Use configurable weights
                    combined_similarity = (
                            self.weights['tfidf'] * tfidf_sim +
                            self.weights['fingerprint'] * fingerprint_sim +
                            self.weights['ngram'] * ngram_sim
                    )

                    logger.debug(f"  Similarities - TF-IDF: {tfidf_sim:.3f}, "
                                 f"Fingerprint: {fingerprint_sim:.3f}, N-gram: {ngram_sim:.3f}, "
                                 f"Combined: {combined_similarity:.3f}")

                    flagged = combined_similarity >= self.threshold
                    if flagged:
                        logger.info(f"FLAGGED: {student1} vs {student2} - similarity: {combined_similarity:.3f}")

                    results.append({
                        'student1_id': metadata[i]['student_id'],
                        'student1_name': metadata[i]['student_name'],
                        'student1_file': metadata[i]['file_name'],
                        'student1_content': metadata[i]['content'],
                        'student2_id': metadata[j]['student_id'],
                        'student2_name': metadata[j]['student_name'],
                        'student2_file': metadata[j]['file_name'],
                        'student2_content': metadata[j]['content'],
                        'tfidf_similarity': float(tfidf_sim),
                        'fingerprint_similarity': float(fingerprint_sim),
                        'ngram_similarity': float(ngram_sim),
                        'similarity': float(combined_similarity),
                        'flagged': flagged,
                        'weights_used': self.weights
                    })

            results.sort(key=lambda x: x['similarity'], reverse=True)
            total_time = time.time() - start_time
            flagged_count = sum(1 for r in results if r['flagged'])

            logger.info(f"Text plagiarism detection completed in {total_time:.2f}s")
            logger.info(f"Generated {len(results)} comparison results, {flagged_count} flagged as potential plagiarism")

            return results

        except Exception as e:
            logger.error(f"Error in text plagiarism detection: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return []

    def _clean_text_content(self, content: str) -> str:
        """Clean and extract meaningful text content."""
        if not content:
            return ""

        # If content looks like HTML, extract text
        if '<' in content and '>' in content:
            try:
                # Remove script and style elements
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)
                # Remove HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)
                # Clean up HTML entities
                content = content.replace('&nbsp;', ' ').replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')
            except Exception as e:
                logger.error(f"Error cleaning HTML content: {e}")

        # Clean up whitespace
        content = re.sub(r'\s+', ' ', content)
        return content.strip()

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based similarity - Improved version."""
        if not text1.strip() or not text2.strip():
            return 0.0

        try:
            # Simplified and more robust TF-IDF
            vectorizer = TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=1,
                token_pattern=r'\b\w+\b',
                strip_accents='unicode'
            )

            documents = [text1, text2]
            tfidf_matrix = vectorizer.fit_transform(documents)

            if tfidf_matrix.shape[1] == 0:
                # Fallback to character-level if no words found
                char_vectorizer = TfidfVectorizer(
                    lowercase=True,
                    ngram_range=(3, 5),
                    analyzer='char',
                    min_df=1
                )
                tfidf_matrix = char_vectorizer.fit_transform(documents)

            if tfidf_matrix.shape[1] > 0:
                similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
                return float(similarity)
            else:
                return 0.0

        except Exception as e:
            logger.error(f"TF-IDF similarity error: {e}")
            # Simple word overlap fallback
            words1 = set(text1.lower().split())
            words2 = set(text2.lower().split())
            if not words1 or not words2:
                return 0.0
            intersection = len(words1 & words2)
            union = len(words1 | words2)
            return intersection / union if union > 0 else 0.0

    def _calculate_fingerprint_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using rolling hash fingerprints."""

        def get_fingerprints(text: str, window_size: int = 5) -> Set[str]:
            fingerprints = set()
            words = text.lower().split()

            if len(words) < window_size:
                return {' '.join(words)}

            for i in range(len(words) - window_size + 1):
                window = ' '.join(words[i:i + window_size])
                fingerprints.add(window)

            return fingerprints

        if not text1.strip() or not text2.strip():
            return 0.0

        fp1 = get_fingerprints(text1)
        fp2 = get_fingerprints(text2)

        if not fp1 or not fp2:
            return 0.0

        intersection = len(fp1.intersection(fp2))
        union = len(fp1.union(fp2))

        return intersection / union if union > 0 else 0.0

    def _calculate_ngram_similarity(self, text1: str, text2: str, n: int = 3) -> float:
        """Calculate character n-gram similarity."""

        def get_ngrams(text: str, n: int) -> Set[str]:
            text = text.lower().replace(' ', '')
            if len(text) < n:
                return {text}
            return set(text[i:i + n] for i in range(len(text) - n + 1))

        if not text1.strip() or not text2.strip():
            return 0.0

        ngrams1 = get_ngrams(text1, n)
        ngrams2 = get_ngrams(text2, n)

        if not ngrams1 or not ngrams2:
            return 0.0

        intersection = len(ngrams1.intersection(ngrams2))
        union = len(ngrams1.union(ngrams2))

        return intersection / union if union > 0 else 0.0


class CodePlagiarismDetector(PlagiarismDetector):
    """Detect plagiarism in code submissions using advanced AST and structural analysis."""

    def __init__(self, threshold: float = 0.8, language: str = 'python',
                 weights: Optional[Dict[str, float]] = None):
        """Initialize with configurable weights."""
        default_weights = {
            'token': 0.25,
            'ast': 0.35,
            'structure': 0.20,
            'normalized': 0.20
        }
        if weights:
            default_weights.update(weights)
        super().__init__(threshold, default_weights)
        self.language = language.lower()

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism in code submissions."""
        if self.language == 'python':
            return self._detect_python_plagiarism(submissions)
        elif self.language in ['c', 'cpp']:
            return self._detect_c_cpp_plagiarism(submissions)
        elif self.language in ['javascript', 'js']:
            return self._detect_javascript_plagiarism(submissions)
        elif self.language == 'java':
            return self._detect_java_plagiarism(submissions)
        else:
            logger.warning(f"Plagiarism detection for {self.language} not implemented yet.")
            return []

    def _detect_python_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced Python plagiarism detection with improved AST analysis."""
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_python_code_advanced(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'content': code_content,
                        'tokens': processed['tokens'],
                        'ast_signature': processed['ast_signature'],
                        'ast_tree': processed['ast_tree'],
                        'structure_hash': processed['structure_hash'],
                        'normalized_code': processed['normalized_code'],
                        'subtree_patterns': processed['subtree_patterns']
                    })
            except Exception as e:
                logger.error(f"Error processing Python code for student {submission.get('student_id', 'Unknown')}: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                # Multiple similarity measures
                token_sim = self._calculate_token_similarity(sub1['tokens'], sub2['tokens'])
                ast_sim = self._calculate_advanced_ast_similarity(
                    sub1['ast_tree'], sub2['ast_tree'],
                    sub1['ast_signature'], sub2['ast_signature'],
                    sub1['subtree_patterns'], sub2['subtree_patterns']
                )
                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.0
                normalized_sim = self._calculate_normalized_similarity(sub1['normalized_code'], sub2['normalized_code'])

                # Use configurable weights
                combined_sim = (
                        self.weights['token'] * token_sim +
                        self.weights['ast'] * ast_sim +
                        self.weights['structure'] * structure_sim +
                        self.weights['normalized'] * normalized_sim
                )

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student1_content': sub1['content'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'student2_content': sub2['content'],
                    'token_similarity': float(token_sim),
                    'ast_similarity': float(ast_sim),
                    'structure_similarity': float(structure_sim),
                    'normalized_similarity': float(normalized_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold,
                    'weights_used': self.weights
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _process_python_code_advanced(self, code: str) -> Dict[str, Any]:
        """Advanced Python code processing with improved AST analysis."""
        try:
            # Tokenization
            tokens = []
            try:
                token_generator = tokenize.generate_tokens(io.StringIO(code).readline)
                for token in token_generator:
                    if token.type not in (tokenize.COMMENT, tokenize.NEWLINE, tokenize.NL,
                                          tokenize.INDENT, tokenize.DEDENT, tokenize.ENCODING, tokenize.ENDMARKER):
                        tokens.append((tokenize.tok_name[token.type], token.string))
            except tokenize.TokenError:
                pass

            # Parse AST
            tree = ast.parse(code)

            # Extract advanced AST features
            ast_signature = self._extract_ast_signature(tree)
            subtree_patterns = self._extract_subtree_patterns(tree)
            structure_hash = self._compute_structure_hash(tree)

            # Create improved normalized code
            normalized_code = self._normalize_python_code_advanced(tree)

            return {
                'tokens': tokens,
                'ast_signature': ast_signature,
                'ast_tree': tree,
                'structure_hash': structure_hash,
                'normalized_code': normalized_code,
                'subtree_patterns': subtree_patterns
            }
        except Exception as e:
            logger.error(f"Error processing Python code: {e}")
            return None

    def _extract_ast_signature(self, tree: ast.AST) -> List[str]:
        """Extract structural signature from AST."""
        signature = []

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                # Include function structure with more detail
                sig = f"Func[name={node.name},args={len(node.args.args)},returns={bool(node.returns)}]"
                # Add control flow inside function
                control_flow = []
                for child in ast.walk(node):
                    if isinstance(child, ast.If):
                        control_flow.append("If")
                    elif isinstance(child, ast.For):
                        control_flow.append("For")
                    elif isinstance(child, ast.While):
                        control_flow.append("While")
                    elif isinstance(child, ast.Try):
                        control_flow.append("Try")
                sig += f"[{','.join(control_flow)}]"
                signature.append(sig)

            elif isinstance(node, ast.ClassDef):
                methods = [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
                signature.append(f"Class[name={node.name},methods={len(methods)}]")

            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                signature.append(f"Import[{type(node).__name__}]")

        return signature

    def _extract_subtree_patterns(self, tree: ast.AST, max_depth: int = 3) -> Set[str]:
        """Extract subtree patterns for structural comparison."""
        patterns = set()

        def extract_pattern(node: ast.AST, depth: int = 0) -> str:
            if depth >= max_depth:
                return "..."

            pattern = type(node).__name__

            # Add specific attributes for certain node types
            if isinstance(node, ast.BinOp):
                pattern += f"({type(node.op).__name__})"
            elif isinstance(node, ast.Compare):
                ops = [type(op).__name__ for op in node.ops]
                pattern += f"({','.join(ops)})"
            elif isinstance(node, ast.Call):
                pattern += "(Call)"

            # Add children
            children = []
            for child in ast.iter_child_nodes(node):
                child_pattern = extract_pattern(child, depth + 1)
                if child_pattern:
                    children.append(child_pattern)

            if children:
                pattern += f"[{','.join(children[:3])}]"

            return pattern

        # Extract patterns from key nodes
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.If, ast.For, ast.While,
                                 ast.ListComp, ast.DictComp, ast.Lambda)):
                pattern = extract_pattern(node)
                patterns.add(pattern)

        return patterns

    def _compute_structure_hash(self, tree: ast.AST) -> str:
        """Compute a more sophisticated structure hash."""
        elements = []

        # Walk the tree in a deterministic order
        def walk_tree(node: ast.AST, parent_type: str = "Module"):
            node_type = type(node).__name__
            elements.append(f"{parent_type}->{node_type}")

            # Sort children by type for consistency
            children = list(ast.iter_child_nodes(node))
            children.sort(key=lambda x: (type(x).__name__, getattr(x, 'lineno', 0)))

            for child in children:
                walk_tree(child, node_type)

        walk_tree(tree)

        # Create hash from sorted elements
        structure_str = '|'.join(sorted(elements))
        return hashlib.sha256(structure_str.encode()).hexdigest()

    def _normalize_python_code_advanced(self, tree: ast.AST) -> str:
        """Advanced normalization preserving structure while removing identifiers."""

        class Normalizer(ast.NodeTransformer):
            def __init__(self):
                self.var_counter = 0
                self.func_counter = 0
                self.class_counter = 0
                self.var_map = {}
                self.func_map = {}
                self.class_map = {}

            def visit_Name(self, node):
                if node.id not in self.var_map and not self._is_builtin(node.id):
                    self.var_map[node.id] = f"var{self.var_counter}"
                    self.var_counter += 1
                if node.id in self.var_map:
                    node.id = self.var_map[node.id]
                return node

            def visit_FunctionDef(self, node):
                if node.name not in self.func_map:
                    self.func_map[node.name] = f"func{self.func_counter}"
                    self.func_counter += 1
                node.name = self.func_map[node.name]
                # Normalize arguments
                for arg in node.args.args:
                    if arg.arg not in self.var_map:
                        self.var_map[arg.arg] = f"arg{self.var_counter}"
                        self.var_counter += 1
                    arg.arg = self.var_map[arg.arg]
                self.generic_visit(node)
                return node

            def visit_ClassDef(self, node):
                if node.name not in self.class_map:
                    self.class_map[node.name] = f"class{self.class_counter}"
                    self.class_counter += 1
                node.name = self.class_map[node.name]
                self.generic_visit(node)
                return node

            def visit_Constant(self, node):
                # Normalize constants
                if isinstance(node.value, str):
                    node.value = "STR"
                elif isinstance(node.value, (int, float)):
                    node.value = "NUM"
                return node

            def _is_builtin(self, name):
                return name in {'True', 'False', 'None', 'print', 'len', 'range',
                                'int', 'str', 'float', 'list', 'dict', 'set'}

        try:
            # FIX: Make a deep copy of the tree before normalizing
            normalized_tree = copy.deepcopy(tree)
            normalized_tree = Normalizer().visit(normalized_tree)

            # Convert back to string
            if hasattr(ast, 'unparse'):
                return ast.unparse(normalized_tree)
            else:
                return ast.dump(normalized_tree)
        except Exception as e:
            logger.error(f"Error in advanced normalization: {e}")
            # Fallback: return a string representation of the AST
            return ast.dump(tree)

    def _calculate_advanced_ast_similarity(self, tree1: ast.AST, tree2: ast.AST,
                                           sig1: List[str], sig2: List[str],
                                           patterns1: Set[str], patterns2: Set[str]) -> float:
        """Calculate advanced AST similarity using multiple metrics."""
        # Signature similarity (sequence-based)
        sig_sim = self._sequence_similarity(sig1, sig2)

        # Pattern similarity (set-based)
        if patterns1 and patterns2:
            pattern_sim = len(patterns1 & patterns2) / len(patterns1 | patterns2)
        else:
            pattern_sim = 0.0

        # Depth distribution similarity
        depth_sim = self._calculate_depth_similarity(tree1, tree2)

        # Combine metrics
        return 0.4 * sig_sim + 0.4 * pattern_sim + 0.2 * depth_sim

    def _sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate sequence similarity using LCS."""
        if not seq1 or not seq2:
            return 0.0 if (seq1 or seq2) else 1.0

        m, n = len(seq1), len(seq2)
        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return 2 * lcs_length / (m + n)

    def _calculate_depth_similarity(self, tree1: ast.AST, tree2: ast.AST) -> float:
        """Compare the depth distribution of ASTs."""

        def get_depth_distribution(tree: ast.AST) -> Dict[str, int]:
            distribution = defaultdict(int)

            def traverse(node: ast.AST, depth: int = 0):
                node_type = type(node).__name__
                distribution[f"{node_type}@{depth}"] += 1
                for child in ast.iter_child_nodes(node):
                    traverse(child, depth + 1)

            traverse(tree)
            return dict(distribution)

        dist1 = get_depth_distribution(tree1)
        dist2 = get_depth_distribution(tree2)

        # Cosine similarity of distributions
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 1.0

        vec1 = np.array([dist1.get(k, 0) for k in all_keys])
        vec2 = np.array([dist2.get(k, 0) for k in all_keys])

        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)

        if norm1 == 0 or norm2 == 0:
            return 1.0 if norm1 == norm2 else 0.0

        return np.dot(vec1, vec2) / (norm1 * norm2)

    def _detect_c_cpp_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improved C/C++ plagiarism detection."""
        logger.warning("C/C++ detection uses basic parsing. For production use, consider integrating pycparser.")
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_c_cpp_code_improved(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'content': code_content,
                        'structure_hash': processed['structure_hash'],
                        'tokens': processed['tokens'],
                        'functions': processed['functions'],
                        'control_structures': processed['control_structures']
                    })
            except Exception as e:
                logger.error(f"Error processing C/C++ code: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        c_weights = self.weights.copy()
        c_weights.update({'structure': 0.35, 'token': 0.35, 'function': 0.30})

        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.3
                token_sim = self._calculate_sequence_similarity(sub1['tokens'], sub2['tokens'])
                function_sim = self._calculate_function_similarity(sub1['functions'], sub2['functions'])

                combined_sim = (
                        c_weights['structure'] * structure_sim +
                        c_weights['token'] * token_sim +
                        c_weights['function'] * function_sim
                )

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student1_content': sub1['content'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'student2_content': sub2['content'],
                    'structure_similarity': float(structure_sim),
                    'token_similarity': float(token_sim),
                    'function_similarity': float(function_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold,
                    'weights_used': c_weights
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _process_c_cpp_code_improved(self, code: str) -> Dict[str, Any]:
        """Improved C/C++ processing with better pattern matching."""
        # Remove comments
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Improved function extraction
        function_patterns = [
            r'(?:(?:static|inline|extern|virtual|const)?\s+)*(?:[\w:]+\s+\*?\s*)+(\w+)\s*\([^)]*\)\s*(?:const)?\s*\{',
            r'(\w+)\s*\([^)]*\)\s*\{',  # Simple functions
        ]

        functions = []
        for pattern in function_patterns:
            functions.extend(re.findall(pattern, code, re.MULTILINE))

        # Extract control structures with context
        control_structures = []
        for match in re.finditer(r'\b(if|for|while|switch|do)\s*\(', code):
            # Try to find the scope of the control structure
            start = match.start()
            context = code[start:start + 100]  # Get some context
            control_structures.append({
                'type': match.group(1),
                'context': context[:50]  # Store limited context
            })

        # Extract classes and structs
        classes = re.findall(r'\b(?:class|struct)\s+(\w+)', code)

        # Create structure hash with more information
        structure_elements = []
        structure_elements.extend([f"func:{f}" for f in functions])
        structure_elements.extend([f"ctrl:{cs['type']}" for cs in control_structures])
        structure_elements.extend([f"class:{c}" for c in classes])

        structure_hash = hashlib.sha256('|'.join(sorted(structure_elements)).encode()).hexdigest()

        # Improved tokenization
        # Remove string literals and character constants
        code_no_strings = re.sub(r'"[^"]*"', 'STRING', code)
        code_no_strings = re.sub(r"'[^']*'", 'CHAR', code_no_strings)

        # Extract meaningful tokens
        tokens = re.findall(r'\b\w+\b|[{}();,=<>!&|+\-*/]', code_no_strings)

        return {
            'structure_hash': structure_hash,
            'tokens': tokens,
            'functions': functions,
            'control_structures': control_structures
        }

    def _detect_javascript_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improved JavaScript plagiarism detection."""
        logger.warning("JavaScript detection uses pattern matching. Consider integrating a JS parser for production.")
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_javascript_code_improved(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'content': code_content,
                        'structure_hash': processed['structure_hash'],
                        'tokens': processed['tokens'],
                        'functions': processed['functions']
                    })
            except Exception as e:
                logger.error(f"Error processing JavaScript code: {e}")

        return self._compare_generic_code(processed_submissions)

    def _process_javascript_code_improved(self, code: str) -> Dict[str, Any]:
        """Improved JavaScript processing."""
        # Remove comments
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Improved function extraction
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=\s*function\s*\(',
            r'(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'const\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'let\s+(\w+)\s*=\s*\([^)]*\)\s*=>',
            r'var\s+(\w+)\s*=\s*\([^)]*\)\s*=>'
        ]

        functions = []
        for pattern in function_patterns:
            functions.extend(re.findall(pattern, code))

        # Extract classes
        classes = re.findall(r'class\s+(\w+)', code)

        # Extract async/await patterns
        async_patterns = len(re.findall(r'\basync\s+', code))
        await_patterns = len(re.findall(r'\bawait\s+', code))

        # Extract control structures
        structures = []
        for ctrl in ['if', 'for', 'while', 'switch', 'try', 'catch']:
            count = len(re.findall(rf'\b{ctrl}\s*\(', code))
            structures.extend([ctrl] * count)

        # Create structure hash
        structure_elements = (
                [f"func:{f}" for f in functions] +
                [f"class:{c}" for c in classes] +
                structures +
                [f"async:{async_patterns}", f"await:{await_patterns}"]
        )

        structure_hash = hashlib.sha256('|'.join(sorted(structure_elements)).encode()).hexdigest()

        # Tokenization
        code_no_strings = re.sub(r'"[^"]*"', 'STRING', code)
        code_no_strings = re.sub(r"'[^']*'", 'STRING', code_no_strings)
        code_no_strings = re.sub(r'`[^`]*`', 'TEMPLATE', code_no_strings)

        tokens = re.findall(r'\b\w+\b|[{}();,=<>!&|+\-*/]', code_no_strings)

        return {
            'structure_hash': structure_hash,
            'tokens': tokens,
            'functions': functions
        }

    def _detect_java_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Improved Java plagiarism detection."""
        logger.warning("Java detection uses pattern matching. Consider integrating javalang for production.")
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_java_code_improved(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'content': code_content,
                        'structure_hash': processed['structure_hash'],
                        'tokens': processed['tokens'],
                        'methods': processed['methods'],
                        'classes': processed['classes']
                    })
            except Exception as e:
                logger.error(f"Error processing Java code: {e}")

        return self._compare_java_code(processed_submissions)

    def _process_java_code_improved(self, code: str) -> Dict[str, Any]:
        """Improved Java processing."""
        # Remove comments
        code = re.sub(r'//.*?\n', '\n', code)
        code = re.sub(r'/\*.*?\*/', '', code, flags=re.DOTALL)

        # Extract package and imports
        packages = re.findall(r'package\s+([\w.]+);', code)
        imports = re.findall(r'import\s+([\w.*]+);', code)

        # Improved class extraction
        classes = re.findall(r'(?:public|private|protected)?\s*(?:static|final|abstract)?\s*class\s+(\w+)', code)
        interfaces = re.findall(r'interface\s+(\w+)', code)
        enums = re.findall(r'enum\s+(\w+)', code)

        # Improved method extraction
        method_pattern = r'(?:public|private|protected)?\s*(?:static|final|synchronized)?\s*(?:[\w<>\[\]]+)\s+(\w+)\s*\([^)]*\)\s*(?:throws\s+[\w,\s]+)?\s*\{'
        methods = re.findall(method_pattern, code)

        # Extract annotations
        annotations = re.findall(r'@(\w+)', code)

        # Extract control structures
        structures = []
        for ctrl in ['if', 'for', 'while', 'switch', 'try', 'catch', 'finally']:
            count = len(re.findall(rf'\b{ctrl}\s*[\(\{{]', code))
            structures.extend([ctrl] * count)

        # Create comprehensive structure hash
        structure_elements = (
                [f"package:{p}" for p in packages] +
                [f"import:{len(imports)}"] +
                [f"class:{c}" for c in classes] +
                [f"interface:{i}" for i in interfaces] +
                [f"enum:{e}" for e in enums] +
                [f"method:{m}" for m in methods] +
                [f"annotation:{a}" for a in set(annotations)] +
                structures
        )

        structure_hash = hashlib.sha256('|'.join(sorted(structure_elements)).encode()).hexdigest()

        # Improved tokenization
        code_no_strings = re.sub(r'"[^"]*"', 'STRING', code)
        code_no_strings = re.sub(r"'[^']*'", 'CHAR', code_no_strings)

        tokens = re.findall(r'\b\w+\b|[{}();,=<>!&|+\-*/]', code_no_strings)

        return {
            'structure_hash': structure_hash,
            'tokens': tokens,
            'methods': methods,
            'classes': classes + interfaces + enums
        }

    def _compare_generic_code(self, processed_submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generic code comparison with configurable weights."""
        if len(processed_submissions) < 2:
            return []

        results = []
        generic_weights = self.weights.copy()
        generic_weights.update({'structure': 0.4, 'token': 0.3, 'function': 0.3})

        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.2
                token_sim = self._calculate_sequence_similarity(sub1['tokens'], sub2['tokens'])
                function_sim = self._calculate_function_similarity(
                    sub1.get('functions', []),
                    sub2.get('functions', [])
                )

                combined_sim = (
                        generic_weights['structure'] * structure_sim +
                        generic_weights['token'] * token_sim +
                        generic_weights['function'] * function_sim
                )

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student1_content': sub1['content'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'student2_content': sub2['content'],
                    'structure_similarity': float(structure_sim),
                    'token_similarity': float(token_sim),
                    'function_similarity': float(function_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold,
                    'weights_used': generic_weights
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _compare_java_code(self, processed_submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Java-specific code comparison with configurable weights."""
        if len(processed_submissions) < 2:
            return []

        results = []
        java_weights = self.weights.copy()
        java_weights.update({
            'structure': 0.3,
            'token': 0.25,
            'method': 0.25,
            'class': 0.2
        })

        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.2
                token_sim = self._calculate_sequence_similarity(sub1['tokens'], sub2['tokens'])
                method_sim = self._calculate_function_similarity(sub1['methods'], sub2['methods'])
                class_sim = self._calculate_function_similarity(sub1['classes'], sub2['classes'])

                combined_sim = (
                        java_weights['structure'] * structure_sim +
                        java_weights['token'] * token_sim +
                        java_weights['method'] * method_sim +
                        java_weights['class'] * class_sim
                )

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student1_content': sub1['content'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'student2_content': sub2['content'],
                    'structure_similarity': float(structure_sim),
                    'token_similarity': float(token_sim),
                    'method_similarity': float(method_sim),
                    'class_similarity': float(class_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold,
                    'weights_used': java_weights
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _calculate_token_similarity(self, tokens1: List[Tuple[str, str]], tokens2: List[Tuple[str, str]]) -> float:
        """Improved token similarity calculation."""
        # Convert to strings for comparison
        str_tokens1 = [f"{t[0]}:{t[1]}" if isinstance(t, tuple) else str(t) for t in tokens1]
        str_tokens2 = [f"{t[0]}:{t[1]}" if isinstance(t, tuple) else str(t) for t in tokens2]

        if not str_tokens1 and not str_tokens2:
            return 1.0
        if not str_tokens1 or not str_tokens2:
            return 0.0

        # Use both set similarity and sequence similarity
        set1, set2 = set(str_tokens1), set(str_tokens2)
        set_similarity = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

        # Sequence similarity (considers order)
        seq_similarity = self._sequence_similarity(str_tokens1[:100], str_tokens2[:100])  # Limit for performance

        # Combine both metrics
        return 0.6 * set_similarity + 0.4 * seq_similarity

    def _calculate_normalized_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between normalized code strings."""
        if not code1.strip() and not code2.strip():
            return 1.0
        if not code1.strip() or not code2.strip():
            return 0.0

        # Use both exact match and token-based comparison
        if code1 == code2:
            return 1.0

        # Token-based comparison
        tokens1 = code1.split()
        tokens2 = code2.split()

        if not tokens1 or not tokens2:
            return 0.0

        # Calculate longest common subsequence ratio
        lcs_sim = self._lcs_similarity(tokens1, tokens2)

        # Calculate Jaccard similarity
        set1, set2 = set(tokens1), set(tokens2)
        jaccard_sim = len(set1 & set2) / len(set1 | set2) if set1 | set2 else 0.0

        return 0.5 * lcs_sim + 0.5 * jaccard_sim

    def _lcs_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity based on longest common subsequence."""
        if not seq1 or not seq2:
            return 0.0

        m, n = len(seq1), len(seq2)
        # Limit size for performance
        if m > 500 or n > 500:
            seq1 = seq1[:500]
            seq2 = seq2[:500]
            m, n = len(seq1), len(seq2)

        dp = [[0] * (n + 1) for _ in range(m + 1)]

        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i - 1] == seq2[j - 1]:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

        lcs_length = dp[m][n]
        return 2 * lcs_length / (m + n)

    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

        # Jaccard similarity
        set1, set2 = set(seq1), set(seq2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _calculate_function_similarity(self, funcs1: List[str], funcs2: List[str]) -> float:
        """Calculate similarity between function lists."""
        if not funcs1 and not funcs2:
            return 1.0
        if not funcs1 or not funcs2:
            return 0.0

        set1, set2 = set(funcs1), set(funcs2)
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0


class PlagiarismManager:
    """Manage plagiarism detection for different types of submissions."""

    def __init__(self):
        self.detectors: Dict[str, PlagiarismDetector] = {}

    def register_detector(self, submission_type: str, detector: PlagiarismDetector):
        """Register a detector for a specific submission type."""
        self.detectors[submission_type] = detector
        logger.info(f"Registered {detector.__class__.__name__} for {submission_type}")

    def detect_plagiarism(self, submission_type: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect plagiarism for a given submission type."""
        if submission_type not in self.detectors:
            logger.error(f"No detector registered for submission type '{submission_type}'")
            return {
                'total_submissions': len(submissions),
                'total_comparisons': 0,
                'flagged_pairs': 0,
                'flagged_student_ids': [],
                'threshold': 0.0,
                'results': [],
                'error': f"No detector registered for {submission_type}"
            }

        detector = self.detectors[submission_type]
        logger.info(f"Starting plagiarism detection for {submission_type} with {len(submissions)} submissions")

        comparison_results = detector.detect(submissions)
        report = detector.generate_report(comparison_results, total_submissions_compared=len(submissions))

        logger.info(f"Detection complete. Found {report['flagged_pairs']} flagged pairs")
        return report


# Example usage and configuration
if __name__ == "__main__":
    # Create manager
    manager = PlagiarismManager()

    # Register detectors with custom weights
    text_detector = TextPlagiarismDetector(
        threshold=0.8,
        weights={'tfidf': 0.5, 'fingerprint': 0.3, 'ngram': 0.2}
    )
    manager.register_detector('text', text_detector)

    python_detector = CodePlagiarismDetector(
        threshold=0.85,
        language='python',
        weights={'token': 0.2, 'ast': 0.4, 'structure': 0.2, 'normalized': 0.2}
    )
    manager.register_detector('python', python_detector)

    # Example submissions
    test_submissions = [
        {
            'student_id': '001',
            'student_name': 'Alice',
            'file_name': 'assignment1.py',
            'content': '''
def calculate_sum(numbers):
    total = 0
    for num in numbers:
        if num > 0:
            total += num
    return total
'''
        },
        {
            'student_id': '002',
            'student_name': 'Bob',
            'file_name': 'assignment1.py',
            'content': '''
def compute_total(items):
    result = 0
    for item in items:
        if item > 0:
            result += item
    return result
'''
        }
    ]

    # Detect plagiarism
    report = manager.detect_plagiarism('python', test_submissions)

    # Print results
    print(f"Total comparisons: {report['total_comparisons']}")
    print(f"Flagged pairs: {report['flagged_pairs']}")
    if report['flagged_pairs'] > 0:
        for result in report['results']:
            if result['flagged']:
                print(f"\nFlagged: {result['student1_name']} vs {result['student2_name']}")
                print(f"  Overall similarity: {result['similarity']:.3f}")
                print(f"  Token similarity: {result['token_similarity']:.3f}")
                print(f"  AST similarity: {result['ast_similarity']:.3f}")
                print(f"  Structure similarity: {result['structure_similarity']:.3f}")
                print(f"  Normalized similarity: {result['normalized_similarity']:.3f}")