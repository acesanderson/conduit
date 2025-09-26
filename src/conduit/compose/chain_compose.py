#!/usr/bin/env python3
"""
ChainML to Mermaid DAG Converter

Converts ChainML workflow files to Mermaid flowchart diagrams showing the DAG structure.

Usage:
    python chainml_to_mermaid.py workflow.chainml
    python chainml_to_mermaid.py workflow.chainml --output diagram.mmd
    python chainml_to_mermaid.py workflow.chainml --style detailed
"""

import yaml
import argparse
import sys
from typing import Dict, List, Set, Optional
from pathlib import Path


class ChainMLToMermaid:
    """Converts ChainML workflows to Mermaid flowchart diagrams"""

    def __init__(self, style: str = "simple"):
        self.style = style  # "simple", "detailed", or "full"
        self.nodes: Dict[str, Dict] = {}
        self.edges: List[tuple] = []
        self.inputs: Dict[str, Dict] = {}
        self.outputs: Dict[str, Dict] = {}
        self.workflow_name = ""

    def parse_chainml(self, chainml_content: str):
        """Parse ChainML content and extract DAG structure"""
        try:
            data = yaml.safe_load(chainml_content)
            workflow = data.get("workflow", {})

            # Extract metadata
            self.workflow_name = workflow.get("name", "Unnamed Workflow")

            # Extract inputs
            self.inputs = workflow.get("inputs", {})

            # Extract outputs
            self.outputs = workflow.get("outputs", {})

            # Extract steps and build DAG
            steps = workflow.get("steps", {})
            self._parse_steps(steps)

        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in ChainML file: {e}")
        except Exception as e:
            raise ValueError(f"Error parsing ChainML: {e}")

    def _parse_steps(self, steps: Dict[str, Dict]):
        """Parse workflow steps and extract dependencies"""
        for step_name, step_config in steps.items():
            # Clean step name (remove 'step ' prefix if present)
            clean_name = step_name.replace("step ", "").strip()

            # Store step information
            self.nodes[clean_name] = {
                "name": clean_name,
                "model": step_config.get("model", "unknown"),
                "description": step_config.get("description", ""),
                "has_condition": "condition" in step_config,
                "has_parser": "parser" in step_config,
                "parser_type": (
                    step_config.get("parser", {}).get("type", "string")
                    if "parser" in step_config
                    else "string"
                ),
            }

            # Extract dependencies and create edges
            depends_on = step_config.get("depends_on", [])
            for dependency in depends_on:
                self.edges.append((dependency, clean_name))

    def _sanitize_node_id(self, name: str) -> str:
        """Convert node name to valid Mermaid ID"""
        # Replace spaces and special characters with underscores
        return name.replace(" ", "_").replace("-", "_").replace(".", "_")

    def _get_node_style(self, node_info: Dict) -> str:
        """Determine node styling based on step properties"""
        if self.style == "simple":
            return ""

        styles = []

        # Color based on model type
        model = node_info["model"].lower()
        if "gpt" in model or "openai" in model:
            styles.append("fill:#e1f5fe")  # Light blue for OpenAI
        elif "claude" in model or "anthropic" in model:
            styles.append("fill:#f3e5f5")  # Light purple for Anthropic
        elif "gemini" in model or "google" in model:
            styles.append("fill:#e8f5e8")  # Light green for Google
        else:
            styles.append("fill:#fff3e0")  # Light orange for others

        # Border style for special properties
        if node_info["has_condition"]:
            styles.append(
                "stroke:#ff9800,stroke-width:3px,stroke-dasharray: 5 5"
            )  # Dashed for conditional
        elif node_info["has_parser"]:
            styles.append(
                "stroke:#4caf50,stroke-width:2px"
            )  # Green for structured output

        return (
            f"style {self._sanitize_node_id(node_info['name'])} {','.join(styles)}"
            if styles
            else ""
        )

    def _format_node_label(self, node_info: Dict) -> str:
        """Format node label based on style setting"""
        name = node_info["name"]

        if self.style == "simple":
            return name
        elif self.style == "detailed":
            model = node_info["model"]
            # Shorten common model names
            model_short = model.replace("gpt-4o-mini", "GPT-4o-mini").replace(
                "claude-3-5-haiku", "Claude-Haiku"
            )
            return f"{name}<br/><small>{model_short}</small>"
        else:  # full
            model = node_info["model"]
            description = (
                node_info["description"][:50] + "..."
                if len(node_info["description"]) > 50
                else node_info["description"]
            )
            parser_info = (
                f"<br/><small>Parser: {node_info['parser_type']}</small>"
                if node_info["has_parser"]
                else ""
            )
            condition_info = (
                "<br/><small>⚠️ Conditional</small>"
                if node_info["has_condition"]
                else ""
            )
            return f"{name}<br/><small>{model}</small><br/><small>{description}</small>{parser_info}{condition_info}"

    def _add_input_output_nodes(self) -> str:
        """Add input and output nodes to the diagram"""
        lines = []

        # Add input nodes
        if self.inputs:
            for input_name in self.inputs.keys():
                input_id = f"input_{self._sanitize_node_id(input_name)}"
                lines.append(f"    {input_id}[📥 {input_name}]")
                if self.style != "simple":
                    lines.append(f"    style {input_id} fill:#e3f2fd,stroke:#1976d2")

        # Add output nodes
        if self.outputs:
            for output_name in self.outputs.keys():
                output_id = f"output_{self._sanitize_node_id(output_name)}"
                lines.append(f"    {output_id}[📤 {output_name}]")
                if self.style != "simple":
                    lines.append(f"    style {output_id} fill:#f1f8e9,stroke:#689f38")

        return "\n".join(lines)

    def _add_input_output_edges(self) -> str:
        """Add edges connecting inputs/outputs to workflow steps"""
        lines = []

        # Connect inputs to steps that use them (simplified - connects to steps with no dependencies)
        if self.inputs:
            root_steps = [
                name
                for name, info in self.nodes.items()
                if not any(edge[1] == name for edge in self.edges)
            ]

            for input_name in self.inputs.keys():
                input_id = f"input_{self._sanitize_node_id(input_name)}"
                for step_name in root_steps:
                    step_id = self._sanitize_node_id(step_name)
                    lines.append(f"    {input_id} --> {step_id}")

        # Connect final steps to outputs (simplified - connects from steps with no dependents)
        if self.outputs:
            final_steps = [
                name
                for name, info in self.nodes.items()
                if not any(edge[0] == name for edge in self.edges)
            ]

            for output_name in self.outputs.keys():
                output_id = f"output_{self._sanitize_node_id(output_name)}"
                for step_name in final_steps:
                    step_id = self._sanitize_node_id(step_name)
                    lines.append(f"    {step_id} --> {output_id}")

        return "\n".join(lines)

    def generate_mermaid(self) -> str:
        """Generate Mermaid flowchart from parsed ChainML"""
        lines = []

        # Header
        lines.append("---")
        lines.append(f"title: {self.workflow_name}")
        lines.append("---")
        lines.append("flowchart TD")
        lines.append("")

        # Add input/output nodes if they exist
        io_nodes = self._add_input_output_nodes()
        if io_nodes:
            lines.append("    %% Input/Output Nodes")
            lines.append(io_nodes)
            lines.append("")

        # Add step nodes
        lines.append("    %% Workflow Steps")
        for node_name, node_info in self.nodes.items():
            node_id = self._sanitize_node_id(node_name)
            label = self._format_node_label(node_info)

            # Choose node shape based on properties
            if node_info["has_condition"]:
                lines.append(f"    {node_id}{{{label}}}")  # Diamond for conditional
            elif node_info["has_parser"]:
                lines.append(f"    {node_id}[{label}]")  # Rectangle for structured
            else:
                lines.append(f"    {node_id}({label})")  # Rounded for simple

        lines.append("")

        # Add step dependencies
        if self.edges:
            lines.append("    %% Dependencies")
            for source, target in self.edges:
                source_id = self._sanitize_node_id(source)
                target_id = self._sanitize_node_id(target)
                lines.append(f"    {source_id} --> {target_id}")
            lines.append("")

        # Add input/output connections
        io_edges = self._add_input_output_edges()
        if io_edges:
            lines.append("    %% Input/Output Connections")
            lines.append(io_edges)
            lines.append("")

        # Add styling
        if self.style != "simple":
            lines.append("    %% Styling")
            for node_name, node_info in self.nodes.items():
                style = self._get_node_style(node_info)
                if style:
                    lines.append(f"    {style}")

        return "\n".join(lines)

    def validate_dag(self) -> List[str]:
        """Validate that the workflow forms a valid DAG"""
        errors = []

        # Check for self-references
        for source, target in self.edges:
            if source == target:
                errors.append(f"Self-reference detected: {source} depends on itself")

        # Check for cycles using DFS
        visited = set()
        rec_stack = set()

        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)

            # Get all nodes this one points to
            dependents = [target for source, target in self.edges if source == node]

            for dependent in dependents:
                if dependent not in visited:
                    if has_cycle(dependent):
                        return True
                elif dependent in rec_stack:
                    errors.append(f"Cycle detected involving: {node} -> {dependent}")
                    return True

            rec_stack.remove(node)
            return False

        # Check all nodes for cycles
        for node in self.nodes.keys():
            if node not in visited:
                has_cycle(node)

        # Check for orphaned dependencies
        all_referenced = set()
        for source, target in self.edges:
            all_referenced.add(source)
            all_referenced.add(target)

        for referenced in all_referenced:
            if referenced not in self.nodes:
                errors.append(f"Step '{referenced}' is referenced but not defined")

        return errors


def main():
    parser = argparse.ArgumentParser(
        description="Convert ChainML workflow files to Mermaid DAG diagrams",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python chainml_to_mermaid.py workflow.chainml
  python chainml_to_mermaid.py workflow.chainml --output diagram.mmd
  python chainml_to_mermaid.py workflow.chainml --style detailed
  python chainml_to_mermaid.py workflow.chainml --style full --validate
        """,
    )

    parser.add_argument("chainml_file", help="Path to ChainML workflow file")

    parser.add_argument(
        "--output", "-o", help="Output file for Mermaid diagram (default: stdout)"
    )

    parser.add_argument(
        "--style",
        "-s",
        choices=["simple", "detailed", "full"],
        default="detailed",
        help="Diagram style: simple (names only), detailed (+ models), full (+ descriptions)",
    )

    parser.add_argument(
        "--validate",
        "-v",
        action="store_true",
        help="Validate DAG structure and report errors",
    )

    args = parser.parse_args()

    # Read ChainML file
    try:
        chainml_path = Path(args.chainml_file)
        if not chainml_path.exists():
            print(f"Error: File '{args.chainml_file}' not found", file=sys.stderr)
            sys.exit(1)

        with open(chainml_path, "r", encoding="utf-8") as f:
            chainml_content = f.read()

    except Exception as e:
        print(f"Error reading file: {e}", file=sys.stderr)
        sys.exit(1)

    # Convert to Mermaid
    try:
        converter = ChainMLToMermaid(style=args.style)
        converter.parse_chainml(chainml_content)

        # Validate if requested
        if args.validate:
            errors = converter.validate_dag()
            if errors:
                print("DAG Validation Errors:", file=sys.stderr)
                for error in errors:
                    print(f"  - {error}", file=sys.stderr)
                sys.exit(1)
            else:
                print("✅ DAG validation passed", file=sys.stderr)

        # Generate Mermaid diagram
        mermaid_output = converter.generate_mermaid()

        # Output to file or stdout
        if args.output:
            output_path = Path(args.output)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(mermaid_output)
            print(f"Mermaid diagram written to: {output_path}", file=sys.stderr)
        else:
            print(mermaid_output)

    except Exception as e:
        print(f"Error converting ChainML: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
