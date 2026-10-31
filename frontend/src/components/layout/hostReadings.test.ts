import { describe, expect, it } from 'vitest';
import type { HostResources } from '@/api/host';
import { hostReadings, shortGpuName } from './hostReadings';
import { formatReading } from './ResourceStrip';

const GIB = 1024 ** 3;
const UUID_3090 = 'GPU-247aa582-0d1b-e161-8156-983ed1fefc57';
const UUID_3080TI = 'GPU-f47ba814-49a2-603f-3595-275284140251';

const BASE: HostResources = {
  observed_at: '2026-10-19T13:00:00+00:00',
  cpu: { percent: 37.4, logical_cores: 32 },
  memory: { used_bytes: 24 * GIB, total_bytes: 64 * GIB },
  gpus: [
    {
      uuid: UUID_3090,
      name: 'NVIDIA GeForce RTX 3090',
      utilization_percent: 97,
      memory_used_bytes: 15.8 * GIB,
      memory_total_bytes: 24 * GIB,
      temperature_c: 66,
    },
  ],
  unavailable: [],
};

const text = (resources: HostResources | null) =>
  hostReadings(resources).map((reading) => `${reading.label}: ${formatReading(reading)}`);

describe('[hostReadings]', () => {
  it('should render CPU, RAM, and the cards utilisation, VRAM and temperature', () => {
    expect(text(BASE)).toEqual([
      'CPU: 37%',
      'RAM: 24.0/64.0 GB',
      'RTX 3090: 97%',
      'VRAM: 15.8/24.0 GB',
      'Temp: 66°C',
    ]);
  });

  it('should name each of two cards so their readings cannot be read as one', () => {
    const two: HostResources = {
      ...BASE,
      gpus: [
        ...BASE.gpus,
        {
          uuid: UUID_3080TI,
          name: 'NVIDIA GeForce RTX 3080 Ti',
          utilization_percent: null,
          memory_used_bytes: 0.5 * GIB,
          memory_total_bytes: 12 * GIB,
          temperature_c: null,
        },
      ],
    };
    const readings = hostReadings(two);
    expect(readings.map((r) => r.id)).toContain(`vram-${UUID_3080TI}`);
    expect(text(two).slice(2)).toEqual([
      'RTX 3090: 97%',
      'RTX 3090 VRAM: 15.8/24.0 GB',
      'RTX 3090 Temp: 66°C',
      'RTX 3080 Ti: utilisation not readable',
      'RTX 3080 Ti VRAM: 0.5/12.0 GB',
      'RTX 3080 Ti Temp: not readable',
    ]);
    // By UUID, never by index (ADR-008).
    expect(readings.some((r) => /\b(?:gpu|vram|temp)-?[01]\b/.test(r.id))).toBe(false);
  });

  it('should say why there is no GPU in one entry, not a row of empty readings', () => {
    const none: HostResources = {
      ...BASE,
      gpus: [],
      unavailable: [{ metric: 'gpu', code: 'nvidia_smi_unavailable', message: 'no GPU visible' }],
    };
    expect(text(none)).toEqual(['CPU: 37%', 'RAM: 24.0/64.0 GB', 'GPU: no GPU visible']);
  });

  it('should show nothing before the first answer and one entry when the read fails', () => {
    expect(hostReadings(null)).toEqual([]);
    const failed = hostReadings(null, { kind: 'transport', status: 0, cause: 'offline' });
    expect(failed.map((r) => `${r.label}: ${formatReading(r)}`)).toEqual([
      'Host: readings unavailable',
    ]);
  });

  it('should treat a body that is not the shape as unavailable rather than crash', () => {
    expect(hostReadings({} as unknown as HostResources).map((r) => formatReading(r))).toEqual([
      'readings unavailable',
    ]);
  });

  it('should shorten driver names', () => {
    expect(shortGpuName('NVIDIA GeForce RTX 3090')).toBe('RTX 3090');
    expect(shortGpuName('NVIDIA A100-SXM4-80GB')).toBe('A100-SXM4-80GB');
  });
});
