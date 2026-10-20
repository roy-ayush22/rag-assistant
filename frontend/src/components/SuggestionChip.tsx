type SuggestionChipProps = {
  children: string
  onSelect: (suggestion: string) => void
  disabled?: boolean
}

export default function SuggestionChip({ children, onSelect, disabled = false }: SuggestionChipProps) {
  return (
    <button
      type="button"
      className="suggestion-chip"
      onClick={() => onSelect(children)}
      disabled={disabled}
    >
      {children}
    </button>
  )
}
