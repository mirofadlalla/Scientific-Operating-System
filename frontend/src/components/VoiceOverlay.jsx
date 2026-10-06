export default function VoiceOverlay({
  active,
  speaking,
  processing,
  transcript,
  status,
  onStop,
  waveHeights,
}) {
  return (
    <div className={`voice-overlay ${active ? 'active' : ''}`}>
      <div className="voice-orb-wrap">
        <div className={`orb-ring ${speaking ? 'speaking' : ''}`} />
        <div className={`orb-ring ${speaking ? 'speaking' : ''}`} />
        <div className={`orb-ring ${speaking ? 'speaking' : ''}`} />
        <div className={`orb-core ${speaking ? 'speaking' : processing ? 'processing' : ''}`}>
          {processing ? '⚙️' : speaking ? '🎤' : '🤖'}
        </div>
      </div>

      <div className="live-waveform">
        {waveHeights.map((h, i) => (
          <span key={i} style={{ height: h + 'px' }} />
        ))}
      </div>

      <div className={`live-transcript ${!transcript ? 'dim' : ''}`}>
        {transcript || 'Speak now…'}
      </div>
      <div className="live-status">{status}</div>
      <button className="overlay-stop-btn" onClick={onStop}>
        ■ Stop Voice
      </button>
    </div>
  );
}
