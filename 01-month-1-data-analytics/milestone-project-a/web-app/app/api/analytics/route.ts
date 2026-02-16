import { NextResponse } from 'next/server';
import { getStats, getGenderDistribution, getCholesterolRisk } from '@/lib/db';

export async function GET() {
  try {
    const stats = getStats();
    const genderDist = getGenderDistribution();
    const cholRisk = getCholesterolRisk();

    return NextResponse.json({
      stats,
      genderDist,
      cholRisk
    });
  } catch (error) {
    console.error('Failed to fetch analytics:', error);
    return NextResponse.json({ error: 'Failed to fetch analytics' }, { status: 500 });
  }
}
