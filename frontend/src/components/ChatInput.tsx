import { useState } from "react";
import type { FormEvent } from "react";

export type ChatInputProps = {
  onSubmit: (text: string) => void;
  disabled?: boolean;
  placeholder?: string;
};

export default function ChatInput({
  onSubmit,
  disabled = false,
  placeholder = "Ask a question about your documents...",
}: ChatInputProps) {
  const [text, setText] = useState("");

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const prompt = text.trim();
    if (!prompt || disabled) return;
    onSubmit(prompt);
    setText("");
  }

  return (
    <form className="chat-input" onSubmit={handleSubmit}>
      <input
        aria-label="Your question"
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder={placeholder}
        disabled={disabled}
      />
      <button
        className="chat-input__submit"
        type="submit"
        aria-label="Send question"
        disabled={disabled || !text.trim()}
      >
        <svg viewBox="0 0 24 24" aria-hidden="true">
          <path d="M12 19V5m-6 6 6-6 6 6" />
        </svg>
      </button>
    </form>
  );
}
