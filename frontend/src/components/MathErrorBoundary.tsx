import { Component, type ReactNode } from "react";

interface Props {
  children: ReactNode;
  raw?: string;
}

interface State {
  error: boolean;
}

/**
 * Catches KaTeX / math rendering errors and falls back to displaying
 * the raw LaTeX in a styled code block instead of crashing the message.
 */
export class MathErrorBoundary extends Component<Props, State> {
  state: State = { error: false };

  static getDerivedStateFromError(): State {
    return { error: true };
  }

  render() {
    if (this.state.error) {
      return (
        <code
          style={{
            color: "var(--accent-secondary)",
            background: "var(--bg-code)",
            padding: "1px 5px",
            borderRadius: 4,
            fontSize: "0.88em",
            fontFamily: "var(--font-llm-code)",
          }}
        >
          {this.props.raw ?? "⚠ math render error"}
        </code>
      );
    }
    return this.props.children;
  }
}
