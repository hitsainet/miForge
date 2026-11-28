/**
 * `GET /api/v1/host/resources` -> the strip's readings.
 *
 * Pure, so the rules are tested without a React tree:
 *
 * - **CPU** percent and **RAM** used of total, the node's figures.
 * - **Per card, by UUID**: its utilisation under its own short name ("RTX 3090"), its VRAM
 *   used of total and its temperature. With more than one card the VRAM and temperature
 *   labels carry the card's name too, so two cards cannot be read as one. The full name and
 *   UUID are the hover text; no index appears anywhere (ADR-008).
 * - **A source that could not be read is one short entry naming why** ("GPU: no GPU
 *   visible"), not a row of empty readings.
 * - Before the first answer the strip shows nothing; if the read itself fails it shows one
 *   entry saying so.
 */
import { Activity, Cpu, HardDrive, MemoryStick, Thermometer } from 'lucide-react';
import type { ApiFailure } from '@/api/errors';
import type { HostResources } from '@/api/host';
import type { ResourceReading } from './types';

const GIB = 1024 ** 3;

/** "NVIDIA GeForce RTX 3090" -> "RTX 3090". */
export function shortGpuName(name: string): string {
  return name.replace(/^NVIDIA\s+/i, '').replace(/^GeForce\s+/i, '') || name;
}

function reasonFor(resources: HostResources, metric: string): string {
  return resources.unavailable.find((entry) => entry.metric === metric)?.message ?? 'unavailable';
}

function describeFailure(failure: ApiFailure): string {
  switch (failure.kind) {
    case 'refusal':
    case 'service_not_configured':
      return failure.message;
    case 'malformed':
      return failure.reason;
    case 'transport':
      return failure.cause;
  }
}

/**
 * Whether a 200 body is the shape this reads. A proxy's `{}` or a changed route must render as
 * "readings unavailable", not crash the shell that every screen sits in.
 */
function isHostResources(value: unknown): value is HostResources {
  if (typeof value !== 'object' || value === null) return false;
  const body = value as Record<string, unknown>;
  return (
    'cpu' in body &&
    'memory' in body &&
    Array.isArray(body['gpus']) &&
    Array.isArray(body['unavailable'])
  );
}

export function hostReadings(
  value: HostResources | null,
  failure: ApiFailure | null = null,
): readonly ResourceReading[] {
  const resources = isHostResources(value) ? value : null;
  if (resources === null) {
    if (value !== null) {
      return [
        {
          id: 'host',
          label: 'Host',
          icon: Cpu,
          value: null,
          unit: '',
          reason: 'readings unavailable',
          title: 'the host readings response was not in the expected shape',
        },
      ];
    }
    if (failure === null) return [];
    return [
      {
        id: 'host',
        label: 'Host',
        icon: Cpu,
        value: null,
        unit: '',
        reason: 'readings unavailable',
        title: describeFailure(failure),
      },
    ];
  }

  const readings: ResourceReading[] = [];
  readings.push(
    resources.cpu === null
      ? {
          id: 'cpu',
          label: 'CPU',
          icon: Cpu,
          value: null,
          unit: '%',
          reason: reasonFor(resources, 'cpu'),
        }
      : {
          id: 'cpu',
          label: 'CPU',
          icon: Cpu,
          value: resources.cpu.percent,
          unit: '%',
          ...(resources.cpu.logical_cores === null
            ? {}
            : { title: `${resources.cpu.logical_cores} logical cores` }),
        },
  );
  readings.push(
    resources.memory === null
      ? {
          id: 'ram',
          label: 'RAM',
          icon: MemoryStick,
          value: null,
          unit: 'GB',
          reason: reasonFor(resources, 'memory'),
        }
      : {
          id: 'ram',
          label: 'RAM',
          icon: MemoryStick,
          value: resources.memory.used_bytes / GIB,
          of: resources.memory.total_bytes / GIB,
          unit: 'GB',
        },
  );

  if (resources.gpus.length === 0) {
    readings.push({
      id: 'gpu',
      label: 'GPU',
      icon: Activity,
      value: null,
      unit: '%',
      reason: reasonFor(resources, 'gpu'),
    });
    return readings;
  }

  const several = resources.gpus.length > 1;
  for (const card of resources.gpus) {
    const name = shortGpuName(card.name);
    const suffix = several ? `-${card.uuid}` : '';
    const prefix = several ? `${name} ` : '';
    const title = `${card.name} · ${card.uuid}`;
    readings.push({
      id: `gpu${suffix}`,
      label: name,
      icon: Activity,
      value: card.utilization_percent,
      unit: '%',
      reason: 'utilisation not readable',
      title,
    });
    readings.push(
      card.memory_used_bytes === null || card.memory_total_bytes === null
        ? {
            id: `vram${suffix}`,
            label: `${prefix}VRAM`,
            icon: HardDrive,
            value: null,
            unit: 'GB',
            reason: 'not readable',
            title,
          }
        : {
            id: `vram${suffix}`,
            label: `${prefix}VRAM`,
            icon: HardDrive,
            value: card.memory_used_bytes / GIB,
            of: card.memory_total_bytes / GIB,
            unit: 'GB',
            title,
          },
    );
    readings.push({
      id: `temp${suffix}`,
      label: `${prefix}Temp`,
      icon: Thermometer,
      value: card.temperature_c,
      unit: '°C',
      precision: 0,
      reason: 'not readable',
      title,
    });
  }
  return readings;
}
