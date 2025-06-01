import os
import ast
import tokenize
import io
import re
import hashlib
import time
import logging
from typing import List, Dict, Any, Tuple, Set
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Set up logging
logger = logging.getLogger('plagiarism_detection')


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
    """Detect plagiarism in text submissions using advanced NLP techniques."""

    def __init__(self, threshold: float = 0.8):
        super().__init__(threshold)
        self.fingerprint_size = 10  # Size of rolling hash fingerprints

    def detect(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism between text submissions using multiple techniques."""
        start_time = time.time()
        logger.info(f"Starting text plagiarism detection with {len(submissions)} submissions")
        logger.debug(f"Threshold: {self.threshold}")

        texts = []
        metadata = []

        for i, submission in enumerate(submissions):
            logger.debug(f"Processing submission {i + 1}: {submission.get('student_name', 'Unknown')}")
            content = submission.get('content')
            if content is None:
                logger.warning(f"Content missing for submission from student {submission.get('student_id', 'Unknown')}")
                continue

            # Clean and extract text content
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
                'file_name': submission['file_name']
            })
            logger.debug(f"Added {submission['student_name']} to analysis (content length: {len(cleaned_content)})")

        logger.info(f"Processing {len(texts)} text submissions for plagiarism detection after filtering")

        if len(texts) < 2:
            logger.warning("Not enough submissions for comparison")
            return []

        try:
            results = []
            total_comparisons = (len(texts) * (len(texts) - 1)) // 2
            logger.info(f"Will perform {total_comparisons} pairwise comparisons")

            comparison_count = 0
            for i in range(len(texts)):
                for j in range(i + 1, len(texts)):
                    comparison_count += 1
                    student1 = metadata[i]['student_name']
                    student2 = metadata[j]['student_name']

                    logger.debug(f"Comparison {comparison_count}/{total_comparisons}: {student1} vs {student2}")

                    # Multiple similarity measures
                    tfidf_start = time.time()
                    tfidf_sim = self._calculate_tfidf_similarity(texts[i], texts[j])
                    tfidf_time = time.time() - tfidf_start

                    fingerprint_start = time.time()
                    fingerprint_sim = self._calculate_fingerprint_similarity(texts[i], texts[j])
                    fingerprint_time = time.time() - fingerprint_start

                    ngram_start = time.time()
                    ngram_sim = self._calculate_ngram_similarity(texts[i], texts[j])
                    ngram_time = time.time() - ngram_start

                    # Weighted combination of similarities
                    combined_similarity = (0.4 * tfidf_sim + 0.3 * fingerprint_sim + 0.3 * ngram_sim)

                    logger.debug(f"  TF-IDF: {tfidf_sim:.3f} ({tfidf_time:.2f}s)")
                    logger.debug(f"  Fingerprint: {fingerprint_sim:.3f} ({fingerprint_time:.2f}s)")
                    logger.debug(f"  N-gram: {ngram_sim:.3f} ({ngram_time:.2f}s)")
                    logger.debug(f"  Combined: {combined_similarity:.3f}")

                    flagged = combined_similarity >= self.threshold
                    if flagged:
                        logger.info(f"FLAGGED: {student1} vs {student2} - similarity: {combined_similarity:.3f}")

                    results.append({
                        'student1_id': metadata[i]['student_id'],
                        'student1_name': metadata[i]['student_name'],
                        'student1_file': metadata[i]['file_name'],
                        'student2_id': metadata[j]['student_id'],
                        'student2_name': metadata[j]['student_name'],
                        'student2_file': metadata[j]['file_name'],
                        'tfidf_similarity': float(tfidf_sim),
                        'fingerprint_similarity': float(fingerprint_sim),
                        'ngram_similarity': float(ngram_sim),
                        'similarity': float(combined_similarity),
                        'flagged': flagged
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
        logger.debug(f"Cleaning text content of length {len(content)}")

        if not content:
            logger.debug("Empty content provided")
            return ""

        # If content looks like HTML, extract text
        if '<' in content and '>' in content:
            logger.debug("Content appears to be HTML, extracting text")
            try:
                import re
                original_length = len(content)

                # Remove script and style elements
                content = re.sub(r'<script[^>]*>.*?</script>', '', content, flags=re.DOTALL | re.IGNORECASE)
                content = re.sub(r'<style[^>]*>.*?</style>', '', content, flags=re.DOTALL | re.IGNORECASE)

                # Remove HTML tags
                content = re.sub(r'<[^>]+>', ' ', content)

                # Clean up HTML entities
                content = content.replace('&nbsp;', ' ')
                content = content.replace('&lt;', '<')
                content = content.replace('&gt;', '>')
                content = content.replace('&amp;', '&')

                logger.debug(f"HTML cleaning: {original_length} -> {len(content)} characters")

            except Exception as e:
                logger.error(f"Error cleaning HTML content: {e}")

        # Clean up whitespace
        import re
        content = re.sub(r'\s+', ' ', content)
        content = content.strip()

        logger.debug(f"Final cleaned content length: {len(content)}")
        if len(content) > 0:
            logger.debug(f"Content preview: {content[:100]}...")

        return content

    def _calculate_tfidf_similarity(self, text1: str, text2: str) -> float:
        """Calculate TF-IDF based similarity."""
        logger.debug("Calculating TF-IDF similarity")

        if not text1.strip() or not text2.strip():
            logger.debug("One or both texts are empty")
            return 0.0

        try:
            vectorizer = TfidfVectorizer(
                lowercase=True,
                stop_words='english',
                ngram_range=(1, 2),  # Use 1-2 grams instead of 1-3
                max_df=0.85,
                min_df=1,
                max_features=1000  # Limit features to prevent memory issues
            )

            logger.debug("Creating TF-IDF matrix")
            tfidf_matrix = vectorizer.fit_transform([text1, text2])
            logger.debug(f"TF-IDF matrix shape: {tfidf_matrix.shape}")

            if tfidf_matrix.shape[1] == 0:
                logger.warning("TF-IDF matrix has no features")
                return 0.0

            similarity = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
            logger.debug(f"TF-IDF similarity: {similarity}")
            return similarity

        except Exception as e:
            logger.error(f"TF-IDF similarity error: {e}")
            return 0.0

    def _calculate_fingerprint_similarity(self, text1: str, text2: str) -> float:
        """Calculate similarity using rolling hash fingerprints (Rabin-Karp style)."""

        def get_fingerprints(text: str, window_size: int = 5) -> Set[str]:  # Smaller window
            fingerprints = set()
            words = text.lower().split()

            if len(words) < window_size:
                return {' '.join(words)}  # Return the entire text as one fingerprint

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

    def __init__(self, threshold: float = 0.8, language: str = 'python'):
        super().__init__(threshold)
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
            print(f"Warning: Plagiarism detection for {self.language} not implemented yet.")
            return []

    def _detect_python_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhanced Python plagiarism detection."""
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_python_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'tokens': processed['tokens'],
                        'ast_nodes': processed['ast_nodes'],
                        'structure_hash': processed['structure_hash'],
                        'normalized_code': processed['normalized_code']
                    })
            except Exception as e:
                print(f"Error processing Python code for student {submission.get('student_id', 'Unknown')}: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                # Multiple similarity measures
                token_sim = self._calculate_token_similarity(sub1['tokens'], sub2['tokens'])
                ast_sim = self._calculate_ast_similarity(sub1['ast_nodes'], sub2['ast_nodes'])
                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.0
                normalized_sim = self._calculate_normalized_similarity(sub1['normalized_code'], sub2['normalized_code'])

                # Weighted combination
                combined_sim = (0.3 * token_sim + 0.3 * ast_sim + 0.2 * structure_sim + 0.2 * normalized_sim)

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'token_similarity': float(token_sim),
                    'ast_similarity': float(ast_sim),
                    'structure_similarity': float(structure_sim),
                    'normalized_similarity': float(normalized_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _process_python_code(self, code: str) -> Dict[str, Any]:
        """Enhanced Python code processing."""
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

            # AST processing
            ast_nodes = []
            structure_elements = []

            if code.strip():
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    node_info = {'type': type(node).__name__}

                    # Enhanced node processing
                    if isinstance(node, ast.FunctionDef):
                        node_info['name'] = node.name
                        node_info['args_count'] = len(node.args.args)
                        structure_elements.append(f"func_{len(node.args.args)}")
                    elif isinstance(node, ast.ClassDef):
                        node_info['name'] = node.name
                        structure_elements.append("class")
                    elif isinstance(node, (ast.For, ast.While)):
                        structure_elements.append("loop")
                    elif isinstance(node, ast.If):
                        structure_elements.append("conditional")

                    ast_nodes.append(node_info)

            # Create structure hash
            structure_hash = hashlib.md5('_'.join(sorted(structure_elements)).encode()).hexdigest()

            # Create normalized code (remove variable names, comments, etc.)
            normalized_code = self._normalize_python_code(code)

            return {
                'tokens': tokens,
                'ast_nodes': ast_nodes,
                'structure_hash': structure_hash,
                'normalized_code': normalized_code
            }
        except Exception as e:
            print(f"Error processing Python code: {e}")
            return None

    def _normalize_python_code(self, code: str) -> str:
        """Normalize Python code by removing variable names and other identifiers."""
        try:
            # Simple normalization - replace identifiers with placeholders
            normalized = re.sub(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', 'VAR', code)
            normalized = re.sub(r'\d+', 'NUM', normalized)
            normalized = re.sub(r'["\'].*?["\']', 'STR', normalized)
            normalized = re.sub(r'\s+', ' ', normalized)
            return normalized.strip()
        except:
            return code

    def _detect_c_cpp_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism in C/C++ code using structural analysis."""
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_c_cpp_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'structure_hash': processed['structure_hash'],
                        'tokens': processed['tokens'],
                        'functions': processed['functions']
                    })
            except Exception as e:
                print(f"Error processing C/C++ code: {e}")

        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.0
                token_sim = self._calculate_sequence_similarity(sub1['tokens'], sub2['tokens'])
                function_sim = self._calculate_function_similarity(sub1['functions'], sub2['functions'])

                combined_sim = (0.4 * structure_sim + 0.3 * token_sim + 0.3 * function_sim)

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'structure_similarity': float(structure_sim),
                    'token_similarity': float(token_sim),
                    'function_similarity': float(function_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _process_c_cpp_code(self, code: str) -> Dict[str, Any]:
        """Process C/C++ code for plagiarism detection."""
        # Extract functions
        function_pattern = r'(?:int|void|float|double|char|long|short)\s+(\w+)\s*\([^)]*\)\s*\{'
        functions = re.findall(function_pattern, code)

        # Extract control structures
        structures = []
        structures.extend(['if'] * len(re.findall(r'\bif\s*\(', code)))
        structures.extend(['for'] * len(re.findall(r'\bfor\s*\(', code)))
        structures.extend(['while'] * len(re.findall(r'\bwhile\s*\(', code)))
        structures.extend(['switch'] * len(re.findall(r'\bswitch\s*\(', code)))

        # Create structure hash
        structure_hash = hashlib.md5('_'.join(sorted(structures + functions)).encode()).hexdigest()

        # Simple tokenization
        tokens = re.findall(r'\w+|[{}();,]', code)

        return {
            'structure_hash': structure_hash,
            'tokens': tokens,
            'functions': functions
        }

    def _detect_javascript_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism in JavaScript code."""
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_javascript_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'structure_hash': processed['structure_hash'],
                        'tokens': processed['tokens'],
                        'functions': processed['functions']
                    })
            except Exception as e:
                print(f"Error processing JavaScript code: {e}")

        return self._compare_generic_code(processed_submissions)

    def _process_javascript_code(self, code: str) -> Dict[str, Any]:
        """Process JavaScript code for plagiarism detection."""
        # Extract functions
        function_patterns = [
            r'function\s+(\w+)\s*\(',
            r'(\w+)\s*:\s*function\s*\(',
            r'(\w+)\s*=>\s*'
        ]

        functions = []
        for pattern in function_patterns:
            functions.extend(re.findall(pattern, code))

        # Extract control structures
        structures = []
        structures.extend(['if'] * len(re.findall(r'\bif\s*\(', code)))
        structures.extend(['for'] * len(re.findall(r'\bfor\s*\(', code)))
        structures.extend(['while'] * len(re.findall(r'\bwhile\s*\(', code)))
        structures.extend(['switch'] * len(re.findall(r'\bswitch\s*\(', code)))

        structure_hash = hashlib.md5('_'.join(sorted(structures + functions)).encode()).hexdigest()
        tokens = re.findall(r'\w+|[{}();,]', code)

        return {
            'structure_hash': structure_hash,
            'tokens': tokens,
            'functions': functions
        }

    def _detect_java_plagiarism(self, submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect plagiarism in Java code."""
        processed_submissions = []

        for submission in submissions:
            code_content = submission.get('content')
            if code_content is None:
                continue

            try:
                processed = self._process_java_code(code_content)
                if processed:
                    processed_submissions.append({
                        'student_id': submission['student_id'],
                        'student_name': submission['student_name'],
                        'file_name': submission['file_name'],
                        'structure_hash': processed['structure_hash'],
                        'tokens': processed['tokens'],
                        'methods': processed['methods'],
                        'classes': processed['classes']
                    })
            except Exception as e:
                print(f"Error processing Java code: {e}")

        return self._compare_java_code(processed_submissions)

    def _process_java_code(self, code: str) -> Dict[str, Any]:
        """Process Java code for plagiarism detection."""
        # Extract classes
        classes = re.findall(r'class\s+(\w+)', code)

        # Extract methods
        method_pattern = r'(?:public|private|protected)?\s*(?:static)?\s*(?:\w+)\s+(\w+)\s*\([^)]*\)\s*\{'
        methods = re.findall(method_pattern, code)

        # Extract control structures
        structures = []
        structures.extend(['if'] * len(re.findall(r'\bif\s*\(', code)))
        structures.extend(['for'] * len(re.findall(r'\bfor\s*\(', code)))
        structures.extend(['while'] * len(re.findall(r'\bwhile\s*\(', code)))
        structures.extend(['switch'] * len(re.findall(r'\bswitch\s*\(', code)))

        structure_hash = hashlib.md5('_'.join(sorted(structures + methods + classes)).encode()).hexdigest()
        tokens = re.findall(r'\w+|[{}();,]', code)

        return {
            'structure_hash': structure_hash,
            'tokens': tokens,
            'methods': methods,
            'classes': classes
        }

    def _compare_generic_code(self, processed_submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generic code comparison for non-Python languages."""
        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.0
                token_sim = self._calculate_sequence_similarity(sub1['tokens'], sub2['tokens'])
                function_sim = self._calculate_function_similarity(sub1['functions'], sub2['functions'])

                combined_sim = (0.4 * structure_sim + 0.3 * token_sim + 0.3 * function_sim)

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'structure_similarity': float(structure_sim),
                    'token_similarity': float(token_sim),
                    'function_similarity': float(function_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _compare_java_code(self, processed_submissions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Java-specific code comparison."""
        if len(processed_submissions) < 2:
            return []

        results = []
        for i in range(len(processed_submissions)):
            for j in range(i + 1, len(processed_submissions)):
                sub1 = processed_submissions[i]
                sub2 = processed_submissions[j]

                structure_sim = 1.0 if sub1['structure_hash'] == sub2['structure_hash'] else 0.0
                token_sim = self._calculate_sequence_similarity(sub1['tokens'], sub2['tokens'])
                method_sim = self._calculate_function_similarity(sub1['methods'], sub2['methods'])
                class_sim = self._calculate_function_similarity(sub1['classes'], sub2['classes'])

                combined_sim = (0.3 * structure_sim + 0.25 * token_sim + 0.25 * method_sim + 0.2 * class_sim)

                results.append({
                    'student1_id': sub1['student_id'],
                    'student1_name': sub1['student_name'],
                    'student1_file': sub1['file_name'],
                    'student2_id': sub2['student_id'],
                    'student2_name': sub2['student_name'],
                    'student2_file': sub2['file_name'],
                    'structure_similarity': float(structure_sim),
                    'token_similarity': float(token_sim),
                    'method_similarity': float(method_sim),
                    'class_similarity': float(class_sim),
                    'similarity': float(combined_sim),
                    'flagged': combined_sim >= self.threshold
                })

        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results

    def _calculate_token_similarity(self, tokens1: List[Tuple[str, str]], tokens2: List[Tuple[str, str]]) -> float:
        """Calculate token similarity using improved TF-IDF."""
        str_tokens1 = [f"{token_type}:{token_val}" for token_type, token_val in tokens1]
        str_tokens2 = [f"{token_type}:{token_val}" for token_type, token_val in tokens2]

        if not str_tokens1 and not str_tokens2:
            return 1.0
        if not str_tokens1 or not str_tokens2:
            return 0.0

        try:
            vectorizer = TfidfVectorizer(min_df=1)
            corpus = [' '.join(str_tokens1), ' '.join(str_tokens2)]
            tfidf_matrix = vectorizer.fit_transform(corpus)

            if tfidf_matrix.shape[1] == 0:
                return 0.0

            return cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        except:
            # Fallback to Jaccard similarity
            set1, set2 = set(str_tokens1), set(str_tokens2)
            if not set1 and not set2:
                return 1.0
            if not set1 or not set2:
                return 0.0

            intersection = len(set1.intersection(set2))
            union = len(set1.union(set2))
            return intersection / union if union > 0 else 0.0

    def _calculate_ast_similarity(self, nodes1: List[Dict[str, Any]], nodes2: List[Dict[str, Any]]) -> float:
        """Calculate AST similarity using improved vector comparison."""

        def get_node_type_counts(nodes: List[Dict[str, Any]]) -> Dict[str, int]:
            counts = {}
            for node in nodes:
                node_type = node.get('type', 'UnknownNode')
                counts[node_type] = counts.get(node_type, 0) + 1
            return counts

        counts1 = get_node_type_counts(nodes1)
        counts2 = get_node_type_counts(nodes2)

        if not counts1 and not counts2:
            return 1.0
        if not counts1 or not counts2:
            return 0.0

        all_node_types = set(counts1.keys()).union(set(counts2.keys()))
        if not all_node_types:
            return 1.0

        vec1 = np.array([counts1.get(nt, 0) for nt in all_node_types])
        vec2 = np.array([counts2.get(nt, 0) for nt in all_node_types])

        # Use cosine similarity
        dot_product = np.dot(vec1, vec2)
        norm_vec1 = np.linalg.norm(vec1)
        norm_vec2 = np.linalg.norm(vec2)

        if norm_vec1 == 0 or norm_vec2 == 0:
            return 1.0 if norm_vec1 == norm_vec2 else 0.0

        return dot_product / (norm_vec1 * norm_vec2)

    def _calculate_normalized_similarity(self, code1: str, code2: str) -> float:
        """Calculate similarity between normalized code strings."""
        if not code1.strip() and not code2.strip():
            return 1.0
        if not code1.strip() or not code2.strip():
            return 0.0

        # Simple sequence matcher approach
        words1 = code1.split()
        words2 = code2.split()

        if not words1 and not words2:
            return 1.0
        if not words1 or not words2:
            return 0.0

        # Calculate longest common subsequence ratio
        def lcs_length(x, y):
            m, n = len(x), len(y)
            dp = [[0] * (n + 1) for _ in range(m + 1)]

            for i in range(1, m + 1):
                for j in range(1, n + 1):
                    if x[i - 1] == y[j - 1]:
                        dp[i][j] = dp[i - 1][j - 1] + 1
                    else:
                        dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])

            return dp[m][n]

        lcs_len = lcs_length(words1, words2)
        max_len = max(len(words1), len(words2))

        return lcs_len / max_len if max_len > 0 else 0.0

    def _calculate_sequence_similarity(self, seq1: List[str], seq2: List[str]) -> float:
        """Calculate similarity between two sequences."""
        if not seq1 and not seq2:
            return 1.0
        if not seq1 or not seq2:
            return 0.0

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
        self.detectors[submission_type] = detector

    def detect_plagiarism(self, submission_type: str, submissions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect plagiarism for a given submission type."""
        if submission_type not in self.detectors:
            print(f"Error: No detector registered for submission type '{submission_type}'")
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
        comparison_results = detector.detect(submissions)
        report = detector.generate_report(comparison_results, total_submissions_compared=len(submissions))

        return report