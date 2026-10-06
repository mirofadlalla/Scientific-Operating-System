export default function Message({ role, text, variant, streaming, thoughts }) {
  if (role === 'ai_think') {
    return (
      <div className="message-row ai">
        <div className="avatar ai">OS</div>
        <div className="thought-box">
          <div className="th-head">Kernel Processing</div>
          {(thoughts || []).map((t, i) => (
            <div key={i} className="thought-line">{t}</div>
          ))}
        </div>
      </div>
    );
  }

  const isAI = role === 'ai';
  const isUser = role === 'user';
  const roleClass = isAI ? 'ai' : isUser ? 'user' : 'sys';

  // Molecular structure rendering check
  const imgMatch = (text || '').match(
    /!\[(.*?)\]\((https:\/\/pubchem\.ncbi\.nlm\.nih\.gov\/rest\/pug\/compound\/.*?\/PNG.*?)\)/
  );

  return (
    <div className={`message-row ${roleClass}`}>
      <div className={`avatar ${roleClass}`}>
        {isAI ? 'OS' : isUser ? 'U' : '⊙'}
      </div>
      <div className={`bubble ${roleClass} ${variant || ''} ${streaming ? 'streaming' : ''}`}>
        {imgMatch ? (
          <>
            <div className="compound-structure-card">
              <div className="card-badge">🧪 Molecular Structure</div>
              <div className="img-wrap">
                <img
                  src={imgMatch[2]}
                  alt={imgMatch[1]}
                  onError={(e) => { e.target.style.display = 'none'; }}
                />
              </div>
              <div className="card-label">{imgMatch[1]}</div>
            </div>
            {text.replace(imgMatch[0], '').trim() && (
              <div>{text.replace(imgMatch[0], '').trim()}</div>
            )}
          </>
        ) : (
          text
        )}

        {text && role !== 'sys' && (
          <button
            className="copy-btn"
            title="Copy message"
            onClick={(e) => {
              const btn = e.currentTarget;
              const cleanText = text.replace(/!\[.*?\]\(.*?\)/g, '').trim();
              navigator.clipboard.writeText(cleanText);
              btn.innerText = '✓ Copied';
              setTimeout(() => { btn.innerText = '📋 Copy'; }, 2000);
            }}
          >
            📋 Copy
          </button>
        )}
      </div>
    </div>
  );
}
