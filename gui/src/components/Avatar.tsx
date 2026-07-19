interface AvatarProps {
  agentId: string;
  size?: number;
}

export function Avatar({ agentId, size = 32 }: AvatarProps) {
  const isUser = agentId === 'user';
  const agent = !isUser ? window.AGENTS?.[agentId] : null;

  return (
    <div
      style={{
        width: size,
        height: size,
        borderRadius: '50%',
        flexShrink: 0,
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        fontSize: Math.floor(size * 0.44),
        background: isUser ? '#14122e' : agent?.bgColor || '#111',
        border: `2px solid ${isUser ? '#6366f1' : agent?.color || '#333'}`,
      }}
    >
      {isUser ? 'U' : agent?.icon}
    </div>
  );
}