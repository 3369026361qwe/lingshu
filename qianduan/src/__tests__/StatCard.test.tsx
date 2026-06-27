import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import StatCard from '../components/StatCard';

describe('StatCard', () => {
  it('renders title and value', () => {
    render(<StatCard title="📊 沪深300" value="3,856.21" />);
    expect(screen.getByText('📊 沪深300')).toBeInTheDocument();
    expect(screen.getByText('3,856.21')).toBeInTheDocument();
  });

  it('renders prefix and suffix', () => {
    render(
      <StatCard
        title="Test"
        value="100"
        prefix={<span data-testid="prefix">↑</span>}
        suffix={<span data-testid="suffix">+5%</span>}
      />
    );
    expect(screen.getByTestId('prefix')).toBeInTheDocument();
    expect(screen.getByTestId('suffix')).toBeInTheDocument();
  });

  it('renders footer', () => {
    render(<StatCard title="Test" value="100" footer="实时行情" />);
    expect(screen.getByText('实时行情')).toBeInTheDocument();
  });

  it('applies accent color to value', () => {
    render(<StatCard title="Test" value="100" accent="#2ECC71" />);
    const value = screen.getByText('100');
    expect(value).toHaveStyle({ color: '#2ECC71' });
  });
});
