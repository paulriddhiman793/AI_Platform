import type { FileEntry } from '@/types';
import { API_URL } from '@/utils/config';

interface FileCardProps {
  entry: FileEntry;
  onSelectFile?: (entry: FileEntry) => void;
}

export function FileCard({ entry, onSelectFile }: FileCardProps) {
  const a = (window as any).AGENTS?.[entry.agent_id];
  const isUpdate = (entry.version || 1) > 1;
  let projRoot = '';
  if (entry.full_path && entry.agent_id && entry.filename) {
    const normalizedPath = entry.full_path.replace(/\\/g, '/');
    const suffix = `${entry.agent_id}/${entry.filename}`.replace(/\\/g, '/');
    const idx = normalizedPath.lastIndexOf(`/${suffix}`);
    if (idx !== -1) {
      projRoot = normalizedPath.substring(0, idx);
    }
  }
  const downloadUrl = `${API_URL}/workspace/files/${entry.agent_id}/${entry.filename}${projRoot ? `?project_root=${encodeURIComponent(projRoot)}` : ''}`;

  return (
    <div
      onClick={() => onSelectFile && onSelectFile(entry)}
      style={{
        padding: '10px 12px',
        borderRadius: 8,
        background: isUpdate ? '#060a10' : '#060f06',
        border: `1px solid ${isUpdate ? '#1a2a3a' : '#1a3a1a'}`,
        marginBottom: 6,
        cursor: onSelectFile ? 'pointer' : 'default',
        animation: 'mIn .22s ease-out'
      }}
    >
      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 4 }}>
        <span style={{ fontSize: 11 }}>{isUpdate ? 'EDIT' : 'FILE'}</span>
        <span style={{ fontSize: 10, color: a?.color || '#4ade80', fontWeight: 700 }}>{entry.agent_id}</span>
        {isUpdate && (
          <span style={{ fontSize: 9, color: '#60a5fa', background: '#0a1628', border: '1px solid #1a3060', padding: '1px 6px', borderRadius: 4 }}>
            v{entry.version} overwritten
          </span>
        )}
        <a
          href={downloadUrl}
          target="_blank"
          rel="noopener noreferrer"
          onClick={(e) => e.stopPropagation()}
          style={{
            fontSize: 9,
            color: '#38bdf8',
            background: '#0c2035',
            border: '1px solid #0284c7',
            padding: '1px 6px',
            borderRadius: 4,
            marginLeft: 'auto',
            textDecoration: 'none',
            fontWeight: 600
          }}
        >
          OPEN / DOWNLOAD
        </a>
        <span style={{ fontSize: 9, color: '#555555' }}>
          {new Date(entry.timestamp).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })}
        </span>
      </div>
      <div style={{ fontSize: 11, color: isUpdate ? '#60a5fa' : '#4ade80', fontFamily: 'monospace' }}>{entry.filename}</div>
      {entry.full_path && (
        <div style={{ fontSize: 10, color: '#2a3a4a', marginTop: 2, fontFamily: 'monospace', wordBreak: 'break-all' }}>{entry.full_path}</div>
      )}
    </div>
  );
}