import { Avatar } from './Avatar';

interface MessageProps {
  msg: any;
  compact?: boolean;
}

export function Message({ msg, compact = false }: MessageProps) {
  const isUser = msg.from === 'user';
  const agent = !isUser ? window.AGENTS?.[msg.from] : null;
  const tagStyle = msg.tag ? window.TAG_STYLES?.[msg.tag] : null;
  const timestamp = new Date(msg.timestamp).toLocaleTimeString([], {
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });

  return (
    <div
      style={{
        display: 'flex',
        flexDirection: isUser ? 'row-reverse' : 'row',
        gap: 12,
        alignItems: 'flex-start',
        animation: 'mIn .28s ease-out',
      }}
    >
      <Avatar agentId={msg.from} size={compact ? 28 : 32} />
      <div style={{ maxWidth: '80%', display: 'flex', flexDirection: 'column', gap: 5 }}>
        <div
          style={{
            display: 'flex',
            alignItems: 'center',
            gap: 7,
            flexDirection: isUser ? 'row-reverse' : 'row',
          }}
        >
          <span
            style={{
              fontSize: compact ? 10 : 11,
              fontWeight: 700,
              textTransform: 'uppercase',
              color: isUser ? '#818cf8' : agent?.color || '#888',
            }}
          >
            {isUser ? 'You' : agent?.name || msg.from}
          </span>
          {msg.tag && !isUser && tagStyle && (
            <span
              style={{
                fontSize: 9,
                fontWeight: 700,
                padding: '2px 7px',
                borderRadius: 4,
                background: tagStyle.bg,
                color: tagStyle.color,
                border: `1px solid ${tagStyle.border}`,
                letterSpacing: '0.07em',
              }}
            >
              {msg.tag}
            </span>
          )}
          <span style={{ fontSize: 9, color: '#2a2a2a' }}>{timestamp}</span>
        </div>
        <div
          style={{
            padding: '11px 16px',
            borderRadius: isUser ? '14px 3px 14px 14px' : '3px 14px 14px 14px',
            background: isUser ? '#1a1740' : '#0b0b0b',
            border: `1px solid ${
              isUser
                ? '#6366f122'
                : tagStyle
                ? tagStyle.border + '44'
                : '#181818'
            }`,
            color: '#c8c8d0',
            fontSize: 13.5,
            lineHeight: 1.7,
            whiteSpace: 'pre-wrap',
          }}
        >
          {msg.content}
        </div>
      </div>
    </div>
  );
}