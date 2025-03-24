import os
import csv
import json
import datetime
from typing import Dict, List, Any
import pandas as pd
import jinja2


class ReportGenerator:
    """Generate reports for grading and plagiarism results."""

    def __init__(self, output_dir: str = './reports'):
        """Initialize the report generator.

        Args:
            output_dir: Directory to save reports
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)

        # Set up Jinja2 for HTML templating
        self.jinja_env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(os.path.join(os.path.dirname(__file__), '..', 'templates')),
            autoescape=jinja2.select_autoescape(['html', 'xml'])
        )

    def generate_grade_csv(self, assignment_name: str, grading_results: List[Dict[str, Any]]) -> str:
        """Generate a CSV file with grading results formatted for myITS Classroom import.

        Args:
            assignment_name: Name of the assignment
            grading_results: List of grading result dictionaries

        Returns:
            Path to the generated CSV file
        """
        # Format timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{assignment_name}_grades_{timestamp}.csv"
        filepath = os.path.join(self.output_dir, filename)

        # Create the CSV
        with open(filepath, 'w', newline='') as csvfile:
            fieldnames = ['Student ID', 'Student Name', 'Grade', 'Feedback']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            writer.writeheader()
            for result in grading_results:
                writer.writerow({
                    'Student ID': result['student_id'],
                    'Student Name': result['student_name'],
                    'Grade': result['grade']['points'],
                    'Feedback': result['grade']['feedback']
                })

        return filepath

    def generate_plagiarism_html(self, assignment_name: str, plagiarism_report: Dict[str, Any]) -> str:
        """Generate an HTML report of plagiarism detection results.

        Args:
            assignment_name: Name of the assignment
            plagiarism_report: Plagiarism report dictionary

        Returns:
            Path to the generated HTML file
        """
        # Format timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{assignment_name}_plagiarism_{timestamp}.html"
        filepath = os.path.join(self.output_dir, filename)

        # Get the template
        try:
            template = self.jinja_env.get_template('plagiarism_report.html')
        except jinja2.exceptions.TemplateNotFound:
            # If template doesn't exist, create a basic one
            template_string = """
            <!DOCTYPE html>
            <html>
            <head>
                <title>{{ title }}</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 20px; }
                    h1 { color: #333; }
                    table { border-collapse: collapse; width: 100%; margin-top: 20px; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                    tr:nth-child(even) { background-color: #f2f2f2; }
                    th { background-color: #4CAF50; color: white; }
                    .flagged { background-color: #FFDDDD; }
                    .summary { margin: 20px 0; padding: 10px; background-color: #f5f5f5; border-left: 5px solid #4CAF50; }
                </style>
            </head>
            <body>
                <h1>{{ title }}</h1>

                <div class="summary">
                    <p><strong>Assignment:</strong> {{ assignment_name }}</p>
                    <p><strong>Date:</strong> {{ date }}</p>
                    <p><strong>Total Submissions:</strong> {{ report.total_submissions }}</p>
                    <p><strong>Flagged Pairs:</strong> {{ report.flagged_pairs }}</p>
                    <p><strong>Flagged Students:</strong> {{ report.flagged_students }}</p>
                    <p><strong>Similarity Threshold:</strong> {{ report.threshold * 100 }}%</p>
                </div>

                <h2>Similarity Results</h2>
                <table>
                    <thead>
                        <tr>
                            <th>Student 1</th>
                            <th>Student 2</th>
                            <th>Similarity</th>
                            <th>Details</th>
                        </tr>
                    </thead>
                    <tbody>
                        {% for result in report.results %}
                        <tr {% if result.flagged %}class="flagged"{% endif %}>
                            <td>{{ result.student1_name }} ({{ result.student1_id }})<br>{{ result.student1_file }}</td>
                            <td>{{ result.student2_name }} ({{ result.student2_id }})<br>{{ result.student2_file }}</td>
                            <td>{{ (result.similarity * 100) | round(2) }}%</td>
                            <td>
                                {% if result.token_similarity is defined %}
                                <p>Token similarity: {{ (result.token_similarity * 100) | round(2) }}%</p>
                                <p>AST similarity: {{ (result.ast_similarity * 100) | round(2) }}%</p>
                                {% endif %}
                            </td>
                        </tr>
                        {% endfor %}
                    </tbody>
                </table>
            </body>
            </html>
            """
            template = jinja2.Template(template_string)

        # Render the template
        html_content = template.render(
            title=f"Plagiarism Report - {assignment_name}",
            assignment_name=assignment_name,
            date=datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            report=plagiarism_report
        )

        # Write the HTML file
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(html_content)

        return filepath

    def generate_summary_report(
            self,
            assignment_name: str,
            grading_results: List[Dict[str, Any]],
            plagiarism_report: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """Generate a summary of grading and plagiarism results.

        Args:
            assignment_name: Name of the assignment
            grading_results: List of grading result dictionaries
            plagiarism_report: Plagiarism report dictionary (optional)

        Returns:
            Summary data
        """
        # Calculate grade statistics
        grades = [result['grade']['points'] for result in grading_results]

        if grades:
            avg_grade = sum(grades) / len(grades)
            max_grade = max(grades)
            min_grade = min(grades)
            median_grade = sorted(grades)[len(grades) // 2]
        else:
            avg_grade = max_grade = min_grade = median_grade = 0

        summary = {
            'assignment_name': assignment_name,
            'date': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'total_submissions': len(grading_results),
            'grade_stats': {
                'average': round(avg_grade, 2),
                'median': round(median_grade, 2),
                'max': max_grade,
                'min': min_grade
            }
        }

        # Add plagiarism data if available
        if plagiarism_report:
            summary['plagiarism'] = {
                'flagged_pairs': plagiarism_report['flagged_pairs'],
                'flagged_students': plagiarism_report['flagged_students'],
                'threshold': plagiarism_report['threshold']
            }

        # Format timestamp
        timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{assignment_name}_summary_{timestamp}.json"
        filepath = os.path.join(self.output_dir, filename)

        # Write the summary JSON
        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        return summary