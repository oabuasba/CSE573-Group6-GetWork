const KINDS = {
  error: { symbol: "✗", label: "Error" },
  warning: { symbol: "!", label: "Demo" },
};

export default function Notice({ kind, label, className = "alert", role, id, children }) {
  const { symbol, label: defaultLabel } = KINDS[kind];
  return (
    <div id={id} role={role} className={`${className} alert-${kind}`}>
      <span aria-hidden="true">{symbol}</span> <strong>{label ?? defaultLabel}</strong> {children}
    </div>
  );
}
