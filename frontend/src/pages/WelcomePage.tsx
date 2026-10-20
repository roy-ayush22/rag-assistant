import { useState } from 'react'
import ChatInput from '../components/ChatInput'
import SuggestionChip from '../components/SuggestionChip'
import './WelcomePage.css'

const suggestions = [
  'What is a synchronous generator?',
  'How does a salient-pole rotor differ?',
  'Explain armature reaction',
]

function BrandMark({ small = false }: { small?: boolean }) {
  return (
    <span className={`brand-mark${small ? ' brand-mark--small' : ''}`} aria-hidden="true">
      <svg viewBox="0 0 24 24"><path d="m13.7 1.8-9 11.1h6l-.8 9.3 9.4-12h-6.1l.5-8.4Z" /></svg>
    </span>
  )
}

export default function WelcomePage() {
  const [lastPrompt, setLastPrompt] = useState('')

  return (
    <main className="welcome-page">
      <header className="welcome-brand">
        <BrandMark small />
        <span>EE Assistant</span>
      </header>

      <section className="welcome-center" aria-labelledby="welcome-title">
        <BrandMark />
        <h1 id="welcome-title">Hello!</h1>
        <p>
          I’m your Electrical Engineering assistant. Ask about synchronous
          <br className="desktop-break" /> machines, motors and generators — answers come straight from your
          <br className="desktop-break" /> documents.
        </p>
      </section>

      <section className="welcome-compose" aria-label="Ask a question">
        <div className="suggestions">
          {suggestions.map((suggestion) => (
            <SuggestionChip key={suggestion} onSelect={setLastPrompt}>
              {suggestion}
            </SuggestionChip>
          ))}
        </div>
        <ChatInput onSubmit={setLastPrompt} />
        <p className="compose-note">
          Answers are generated from your indexed documents and cite their sources. Always verify critical values.
        </p>
      </section>
      <span className="sr-only" aria-live="polite">{lastPrompt ? `Question ready: ${lastPrompt}` : ''}</span>
    </main>
  )
}
