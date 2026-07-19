import type { P2PEntry } from '@/types';

interface ActivityCardProps {
  entry: P2PEntry;
}

export function ActivityCard({ entry }: ActivityCardProps) {
  const fa = (window as any).AGENTS?.[entry.from];
  const ta = (window as any).AGENTS?.[entry.to];
  if (!fa || !ta) return null;

  return (
    <div style={{
      padding: '12px 14px',
      borderRadius: 8,
      background: '#090909',
      border: `1px solid ${fa.color}28`,
      marginBottom: 6,
      animation: 'mIn .22s ease-out'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
        <div style={{
          width: 24, height: 24, borderRadius: '50%', background: fa.bgColor,
          border: `2px solid ${fa.color}`, display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 10, fontWeight: 700
        }}>{fa.icon}</div>
        <span style={{ fontSize: 10, color: fa.color, fontWeight: 700 }}>{fa.name}</span>
        <svg width="14" height="10" viewBox="0 0 14 10" fill="none">
          <path d="M1 5h10M8 2l3 3-3 3" stroke="#333" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round"/>
        </svg>
        <div style={{
          width: 24, height: 24, borderRadius: '50%', background: ta.bgColor,
          border: `2px solid ${ta.color}`, display: 'flex', alignItems: 'center',
          justifyContent: 'center', fontSize: 10, fontWeight: 700
        }}>{ta.icon}</div>
        <span style={{ fontSize: 10, color: ta.color, fontWeight: 700 }}>{ta.name}</span>
        <span style={{ fontSize: 9, color: '#252525', marginLeft: 'auto' }}>
          {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
      </div>
      <div style={{ fontSize: 12, color: '#666', lineHeight: 1.6, padding: '8px 10px', borderRadius: 6, background: '#060606', border: '1px solid #141414' }}>
        {entry.content}
      </div>
    </div>
  );
}