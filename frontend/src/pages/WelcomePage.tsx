import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import ChatInput from "../components/ChatInput";
import SuggestionChip from "../components/SuggestionChip";
import "./WelcomePage.css";

type Source = {
  source: string;
  page: number;
  content: string;
  similarity: number;
};

type Message = {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
};

type AskResponse = {
  answer: string;
  sources: Source[];
};

const suggestions = [
  "What is a synchronous machine?",
  "How does a salient-pole rotor differ?",
  "Explain armature reaction",
];

function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <span
      className={`brand-mark${small ? " brand-mark--small" : ""}`}
      aria-hidden="true"
    >
      <svg viewBox="0 0 24 24">
        <path d="m13.7 1.8-9 11.1h6l-.8 9.3 9.4-12h-6.1l.5-8.4Z" />
      </svg>
    </span>
  );
}

export default function WelcomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const conversationEnd = useRef<HTMLDivElement>(null);
  const hasConversation = messages.length > 0;

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isLoading]);

  async function ask(question: string) {
    const prompt = question.trim();
    if (!prompt || isLoading) return;

    setMessages((current) => [...current, { role: "user", content: prompt }]);
    setIsLoading(true);
    try {
      const response = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question: prompt, top_k: 5 }),
      });
      const payload = (await response.json()) as
        AskResponse | { detail?: string };
      if (!response.ok) {
        throw new Error(
          "detail" in payload && payload.detail
            ? payload.detail
            : "The request could not be completed.",
        );
      }
      const result = payload as AskResponse;
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: result.answer,
          sources: result.sources,
        },
      ]);
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Could not connect to the RAG API.";
      setMessages((current) => [
        ...current,
        {
          role: "assistant",
          content: `I couldn't answer that right now. ${message}`,
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  }

  return (
    <main
      className={`welcome-page${hasConversation ? " welcome-page--chat" : ""}`}
    >
      <header className="welcome-brand">
        <BrandMark small />
        <span>EE Assistant</span>
      </header>

      {!hasConversation ? (
        <section className="welcome-center" aria-labelledby="welcome-title">
          <BrandMark />
          <h1 id="welcome-title">Hello!</h1>
          <p>
            I’m your Electrical Engineering assistant. Ask about synchronous
            <br className="desktop-break" /> machines, motors and generators —
            answers come straight from your
            <br className="desktop-break" /> documents.
          </p>
        </section>
      ) : (
        <section
          className="chat-thread"
          aria-label="Conversation"
          aria-live="polite"
        >
          {messages.map((message, index) => (
            <article
              className={`chat-message chat-message--${message.role}`}
              key={`${index}-${message.role}`}
            >
              <div className="chat-message__label">
                {message.role === "user" ? "You" : "EE Assistant"}
              </div>
              <div className="chat-message__content">
                <ReactMarkdown
                  remarkPlugins={[remarkGfm, remarkMath]}
                  rehypePlugins={[rehypeKatex]}
                  components={{
                    table: (props) => (
                      <div className="table-wrap">
                        <table {...props} />
                      </div>
                    ),
                  }}
                >
                  {message.content}
                </ReactMarkdown>
              </div>
              {message.sources && message.sources.length > 0 && (
                <details className="chat-sources">
                  <summary>Sources ({message.sources.length})</summary>
                  <ol>
                    {message.sources.map((source, sourceIndex) => (
                      <li
                        key={`${source.source}-${source.page}-${sourceIndex}`}
                      >
                        <strong>
                          [{sourceIndex + 1}] {source.source}, page{" "}
                          {source.page}
                        </strong>
                        <p>{source.content}</p>
                      </li>
                    ))}
                  </ol>
                </details>
              )}
            </article>
          ))}
          {isLoading && (
            <p className="chat-loading" role="status">
              Searching your documents and writing an answer…
            </p>
          )}
          <div ref={conversationEnd} />
        </section>
      )}

      <section
        className={`welcome-compose${hasConversation ? " welcome-compose--chat" : ""}`}
        aria-label="Ask a question"
      >
        {!hasConversation && (
          <div className="suggestions">
            {suggestions.map((suggestion) => (
              <SuggestionChip
                key={suggestion}
                onSelect={ask}
                disabled={isLoading}
              >
                {suggestion}
              </SuggestionChip>
            ))}
          </div>
        )}
        <ChatInput onSubmit={ask} disabled={isLoading} />
        <p className="compose-note">
          Answers are generated from your indexed documents and cite their
          sources. Always verify critical values.
        </p>
      </section>
    </main>
  );
}
