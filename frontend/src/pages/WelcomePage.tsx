import { useEffect, useRef, useState } from "react";
import ReactMarkdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkMath from "remark-math";
import remarkGfm from "remark-gfm";
import ChatInput from "../components/ChatInput";
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

type Conversation = {
  id: string;
  title: string;
  messages: Message[];
  updatedAt: number;
};

type AskResponse = {
  answer: string;
  sources: Source[];
};

const CONVERSATIONS_KEY = "ee-assistant-conversations";

function readConversations(): Conversation[] {
  try {
    const saved = localStorage.getItem(CONVERSATIONS_KEY);
    const parsed: unknown = saved ? JSON.parse(saved) : [];
    if (!Array.isArray(parsed)) return [];
    return parsed.filter(
      (item): item is Conversation =>
        typeof item?.id === "string" &&
        typeof item?.title === "string" &&
        Array.isArray(item?.messages) &&
        typeof item?.updatedAt === "number",
    );
  } catch {
    return [];
  }
}

function createConversationId() {
  return globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random()}`;
}

function conversationTitle(question: string) {
  const title = question.trim();
  return title.length > 42 ? `${title.slice(0, 42).trim()}…` : title;
}

function BrandMark() {
  return (
    <span className="brand-mark" aria-hidden="true">
      <svg viewBox="0 0 24 24">
        <path d="m13.7 1.8-9 11.1h6l-.8 9.3 9.4-12h-6.1l.5-8.4Z" />
      </svg>
    </span>
  );
}

export default function WelcomePage() {
  const [conversations, setConversations] =
    useState<Conversation[]>(readConversations);
  const [activeConversationId, setActiveConversationId] = useState<
    string | null
  >(null);
  const [loadingConversationId, setLoadingConversationId] = useState<
    string | null
  >(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const conversationEnd = useRef<HTMLDivElement>(null);

  const activeConversation = conversations.find(
    (conversation) => conversation.id === activeConversationId,
  );
  const messages = activeConversation?.messages ?? [];
  const hasConversation = messages.length > 0;
  const isLoading =
    activeConversationId !== null &&
    loadingConversationId === activeConversationId;
  const recentConversations = [...conversations]
    .filter((conversation) => conversation.messages.length > 0)
    .sort((a, b) => b.updatedAt - a.updatedAt);

  useEffect(() => {
    const saved = conversations.filter(
      (conversation) => conversation.messages.length > 0,
    );
    localStorage.setItem(CONVERSATIONS_KEY, JSON.stringify(saved));
  }, [conversations]);

  useEffect(() => {
    conversationEnd.current?.scrollIntoView({
      behavior: "smooth",
      block: "end",
    });
  }, [messages, isLoading]);

  function appendMessage(conversationId: string, message: Message) {
    setConversations((current) => {
      const existing = current.find(
        (conversation) => conversation.id === conversationId,
      );
      const base: Conversation = existing ?? {
        id: conversationId,
        title:
          message.role === "user"
            ? conversationTitle(message.content)
            : "New chat",
        messages: [],
        updatedAt: Date.now(),
      };
      const title =
        base.messages.length === 0 && message.role === "user"
          ? conversationTitle(message.content)
          : base.title;
      const updated: Conversation = {
        ...base,
        title,
        messages: [...base.messages, message],
        updatedAt: Date.now(),
      };
      return [updated, ...current.filter((item) => item.id !== conversationId)];
    });
  }

  function startNewChat() {
    setActiveConversationId(createConversationId());
    setSidebarOpen(false);
  }

  function deleteConversation(conversationId: string) {
    const remaining = conversations.filter(
      (conversation) => conversation.id !== conversationId,
    );
    setConversations(remaining);
    if (activeConversationId === conversationId) {
      setActiveConversationId(remaining[0]?.id ?? null);
    }
  }

  async function ask(question: string) {
    const prompt = question.trim();
    if (!prompt || isLoading) return;

    const conversationId = activeConversationId ?? createConversationId();
    setActiveConversationId(conversationId);
    appendMessage(conversationId, { role: "user", content: prompt });
    setLoadingConversationId(conversationId);

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
      appendMessage(conversationId, {
        role: "assistant",
        content: result.answer,
        sources: result.sources,
      });
    } catch (error) {
      const message =
        error instanceof Error
          ? error.message
          : "Could not connect to the RAG API.";
      appendMessage(conversationId, {
        role: "assistant",
        content: `I couldn't answer that right now. ${message}`,
      });
    } finally {
      setLoadingConversationId((current) =>
        current === conversationId ? null : current,
      );
    }
  }

  return (
    <main
      className={`welcome-page${hasConversation ? " welcome-page--chat" : ""}`}
    >
      {!sidebarOpen && (
        <button
          className="sidebar-toggle"
          type="button"
          aria-label="Open chat history"
          aria-expanded={false}
          onClick={() => setSidebarOpen(true)}
        >
          <svg viewBox="0 0 24 24" aria-hidden="true">
            <path d="M4 6h16M4 12h16M4 18h16" />
          </svg>
        </button>
      )}

      {sidebarOpen && (
        <>
          <button
            className="sidebar-backdrop"
            type="button"
            aria-label="Close chat history"
            onClick={() => setSidebarOpen(false)}
          />
          <aside className="chat-sidebar" aria-label="Chat history">
            <div className="chat-sidebar__top">
              <h2>Chats</h2>
              <button
                className="sidebar-close"
                type="button"
                aria-label="Close chat history"
                onClick={() => setSidebarOpen(false)}
              >
                ×
              </button>
            </div>
            <button
              className="new-chat-button"
              type="button"
              onClick={startNewChat}
            >
              <span aria-hidden="true">＋</span> New chat
            </button>
            <div className="chat-sidebar__section-title">Recent chats</div>
            <nav className="recent-chats" aria-label="Recent chats">
              {recentConversations.length === 0 ? (
                <p className="recent-chats__empty">
                  Your chats will appear here.
                </p>
              ) : (
                recentConversations.map((conversation) => (
                  <div
                    className={`recent-chat${conversation.id === activeConversationId ? " recent-chat--active" : ""}`}
                    key={conversation.id}
                  >
                    <button
                      className="recent-chat__select"
                      type="button"
                      aria-current={
                        conversation.id === activeConversationId
                          ? "page"
                          : undefined
                      }
                      onClick={() => {
                        setActiveConversationId(conversation.id);
                        setSidebarOpen(false);
                      }}
                      title={conversation.title}
                    >
                      {conversation.title}
                    </button>
                    <button
                      className="recent-chat__delete"
                      type="button"
                      aria-label={`Delete ${conversation.title}`}
                      onClick={() => deleteConversation(conversation.id)}
                    >
                      ×
                    </button>
                  </div>
                ))
              )}
            </nav>
          </aside>
        </>
      )}

      {!hasConversation ? (
        <section className="welcome-center" aria-labelledby="welcome-title">
          <div className="welcome-greeting">
            <BrandMark />
            <h1 id="welcome-title">Hello!</h1>
          </div>
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
        <ChatInput onSubmit={ask} disabled={isLoading} />
      </section>
    </main>
  );
}
