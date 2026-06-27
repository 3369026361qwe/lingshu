import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import RiskBadge from '../components/RiskBadge';

describe('RiskBadge', () => {
  it('renders LOW level in Chinese', () => {
    render(<RiskBadge level="LOW" />);
    expect(screen.getByText('低风险')).toBeInTheDocument();
  });

  it('renders CRITICAL level', () => {
    render(<RiskBadge level="CRITICAL" />);
    expect(screen.getByText('极高风险')).toBeInTheDocument();
  });

  it('renders level code when showLabel is false', () => {
    render(<RiskBadge level="MEDIUM" showLabel={false} />);
    expect(screen.getByText('MEDIUM')).toBeInTheDocument();
  });

  it('renders all 4 levels', () => {
    const { rerender } = render(<RiskBadge level="LOW" />);
    expect(screen.getByText('低风险')).toBeInTheDocument();
    rerender(<RiskBadge level="MEDIUM" />);
    expect(screen.getByText('中风险')).toBeInTheDocument();
    rerender(<RiskBadge level="HIGH" />);
    expect(screen.getByText('高风险')).toBeInTheDocument();
    rerender(<RiskBadge level="CRITICAL" />);
    expect(screen.getByText('极高风险')).toBeInTheDocument();
  });
});
