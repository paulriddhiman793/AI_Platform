import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import { Avatar } from '../components/Avatar';
import { StatusDot } from '../components/StatusDot';

// Mock the AGENTS global
beforeEach(() => {
  vi.resetModules();
  (window as any).AGENTS = {
    ml_engineer: {
      id: 'ml_engineer',
      name: 'ML Engineer',
      shortName: 'ML',
      icon: 'ML',
      color: '#10b981',
      bgColor: '#022c22',
      role: 'Builds ML models',
      status: 'idle',
    },
  };
  (window as any).TAG_STYLES = {
    STATUS: { bg: '#0d1f0d', color: '#4ade80', border: '#1a3a1a' },
  };
});

describe('Avatar', () => {
  it('renders user avatar', () => {
    render(<Avatar agentId="user" size={32} />);
    const avatar = screen.getByText('U');
    expect(avatar).toBeInTheDocument();
    // Check the parent div has the right background (style may be in RGB format)
    const parent = avatar.parentElement;
    expect(parent).toBeInTheDocument();
  });

  it('renders agent avatar', () => {
    render(<Avatar agentId="ml_engineer" size={32} />);
    const avatar = screen.getByText('ML');
    expect(avatar).toBeInTheDocument();
    const parent = avatar.parentElement;
    expect(parent).toBeInTheDocument();
  });
});

describe('StatusDot', () => {
  it('renders idle status', () => {
    render(<StatusDot status="idle" />);
    expect(screen.getByText('Idle')).toBeInTheDocument();
  });

  it('renders working status', () => {
    render(<StatusDot status="working" />);
    expect(screen.getByText('Working')).toBeInTheDocument();
  });

  it('renders error status', () => {
    render(<StatusDot status="error" />);
    expect(screen.getByText('Error')).toBeInTheDocument();
  });
});