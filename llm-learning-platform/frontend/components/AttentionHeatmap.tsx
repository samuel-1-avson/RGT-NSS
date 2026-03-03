'use client';

import React, { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';

interface AttentionHeatmapProps {
  tokens: string[];
  attentionMatrix: number[][];
  width?: number;
  height?: number;
  onCellHover?: (row: number, col: number, value: number) => void;
  onCellClick?: (row: number, col: number) => void;
}

export const AttentionHeatmap: React.FC<AttentionHeatmapProps> = ({
  tokens,
  attentionMatrix,
  width = 500,
  height = 500,
  onCellHover,
  onCellClick,
}) => {
  const svgRef = useRef<SVGSVGElement>(null);
  const [selectedCell, setSelectedCell] = useState<{ row: number; col: number } | null>(null);

  useEffect(() => {
    if (!svgRef.current || tokens.length === 0) return;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();

    const margin = { top: 80, right: 20, bottom: 80, left: 80 };
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    const cellSize = Math.min(
      innerWidth / tokens.length,
      innerHeight / tokens.length
    );

    // Color scale - using a vibrant purple to pink gradient
    const colorScale = d3.scaleSequential(d3.interpolatePlasma)
      .domain([0, 1]);

    const g = svg.append('g')
      .attr('transform', `translate(${margin.left},${margin.top})`);

    // Create cells
    const cells = g.selectAll('.cell')
      .data(attentionMatrix.flatMap((row, i) =>
        row.map((value, j) => ({ i, j, value }))
      ))
      .enter()
      .append('rect')
      .attr('class', 'cell')
      .attr('x', d => d.j * cellSize)
      .attr('y', d => d.i * cellSize)
      .attr('width', cellSize - 1)
      .attr('height', cellSize - 1)
      .attr('fill', d => colorScale(d.value))
      .attr('stroke', 'white')
      .attr('stroke-width', 0.5)
      .attr('rx', 2)
      .style('cursor', 'pointer')
      .on('mouseover', function(event, d) {
        d3.select(this)
          .attr('stroke', '#8b5cf6')
          .attr('stroke-width', 2);
        onCellHover?.(d.i, d.j, d.value);
      })
      .on('mouseout', function() {
        d3.select(this)
          .attr('stroke', 'white')
          .attr('stroke-width', 0.5);
      })
      .on('click', function(event, d) {
        setSelectedCell({ row: d.i, col: d.j });
        onCellClick?.(d.i, d.j);
      });

    // Add token labels (top)
    g.selectAll('.token-label-top')
      .data(tokens)
      .enter()
      .append('text')
      .attr('class', 'token-label-top')
      .attr('x', (_, i) => i * cellSize + cellSize / 2)
      .attr('y', -10)
      .attr('text-anchor', 'middle')
      .attr('transform', (_, i) =>
        `rotate(-45, ${i * cellSize + cellSize / 2}, -10)`)
      .text(d => d)
      .style('font-size', '11px')
      .style('fill', '#94a3b8');

    // Add token labels (left)
    g.selectAll('.token-label-left')
      .data(tokens)
      .enter()
      .append('text')
      .attr('class', 'token-label-left')
      .attr('x', -10)
      .attr('y', (_, i) => i * cellSize + cellSize / 2)
      .attr('text-anchor', 'end')
      .attr('dominant-baseline', 'middle')
      .text(d => d)
      .style('font-size', '11px')
      .style('fill', '#94a3b8');

    // Add color legend
    const legendWidth = 15;
    const legendHeight = 150;
    const legendX = innerWidth + 20;
    const legendY = (innerHeight - legendHeight) / 2;

    // Create gradient
    const defs = svg.append('defs');
    const gradient = defs.append('linearGradient')
      .attr('id', 'legend-gradient')
      .attr('x1', '0%')
      .attr('y1', '100%')
      .attr('x2', '0%')
      .attr('y2', '0%');

    const gradientStops = d3.range(0, 1.01, 0.01);
    gradient.selectAll('stop')
      .data(gradientStops)
      .enter()
      .append('stop')
      .attr('offset', d => `${d * 100}%`)
      .attr('stop-color', d => colorScale(d));

    // Draw legend rectangle
    g.append('rect')
      .attr('x', legendX)
      .attr('y', legendY)
      .attr('width', legendWidth)
      .attr('height', legendHeight)
      .style('fill', 'url(#legend-gradient)')
      .attr('rx', 3);

    // Legend labels
    g.append('text')
      .attr('x', legendX + legendWidth + 8)
      .attr('y', legendY)
      .text('1.0')
      .style('font-size', '10px')
      .style('fill', '#64748b');

    g.append('text')
      .attr('x', legendX + legendWidth + 8)
      .attr('y', legendY + legendHeight)
      .text('0.0')
      .style('font-size', '10px')
      .style('fill', '#64748b');

  }, [tokens, attentionMatrix, width, height, onCellHover, onCellClick]);

  return (
    <div className="relative">
      <svg ref={svgRef} width={width} height={height} className="rounded-lg" />
      {selectedCell && (
        <div className="absolute top-2 right-2 bg-slate-800 text-white text-xs p-2 rounded-lg shadow-lg">
          {tokens[selectedCell.row]} → {tokens[selectedCell.col]}
        </div>
      )}
    </div>
  );
};
