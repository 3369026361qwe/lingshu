import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ScoreBar from '../components/ScoreBar';

describe('ScoreBar', () => {
  it('renders label and percentage', () => {
    render(<ScoreBar label="ROE" value={0.18} />);
    expect(screen.getByText('ROE')).toBeInTheDocument();
    expect(screen.getByText('18%')).toBeInTheDocument();
  });

  it('renders with custom color', () => {
    render(<ScoreBar label="GNN" value={0.22} color="#D4AF37" />);
    const pct = screen.getByText('22%');
    expect(pct).toHaveStyle({ color: '#D4AF37' });
  });

  it('clamps bar width at 100% when value exceeds maxWidth', () => {
    render(<ScoreBar label="High" value={0.30} maxWidth={0.25} />);
    expect(screen.getByText('30%')).toBeInTheDocument();
    // Bar should not exceed container
  });

  it('renders 0% for zero value', () => {
    render(<ScoreBar label="Zero" value={0} />);
    expect(screen.getByText('0%')).toBeInTheDocument();
  });
});
