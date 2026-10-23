/**
 * The Domain packs screen (F7 task 12.3; FR-007.29).
 *
 * Page head with a one-line subtitle, `Import pack` and `New pack` actions, and the
 * `NoBuiltInDomainBanner` - **the only orange panel on the screen** (ADR-012: the callout
 * banner tint is one of the seven permitted accent surfaces, and nothing else here is one).
 *
 * **The server is the truth and there is no optimistic validity** (F7 task 12.2): the listing
 * is re-read after an import, and a pack's validity is whatever
 * `GET /api/v1/packs`'s `validity` block says. A screen that computed validity would be a
 * second answer to a question the backend re-resolves on every use.
 *
 * **Drafts are not packs** and never appear in this listing (F7's E13); the Onboard screen
 * owns them.
 */
import { useCallback, useEffect, useState, type ReactNode } from 'react';
import { PageHead } from '@/components/layout/PageHead';
import { Card } from '@/components/common/Card';
import { Button } from '@/components/common/Button';
import { FailureNotice } from '@/components/common/FailureNotice';
import { useNavigation } from '@/contexts/NavigationContext';
import { fetchPackLineage, fetchPacks, packExportUrl, type SlotLineage } from '@/api/packs';
import type { ApiFailure } from '@/api/errors';
import type { PackListingRow } from '@/types/packs';
import { PackCard } from './PackCard';
import { NoBuiltInDomainBanner } from './NoBuiltInDomainBanner';
import { ImportPackDialog } from './ImportPackDialog';
import { PartIndex } from './PartIndex';

export function PacksPage(): ReactNode {
  const navigation = useNavigation();
  const [packs, setPacks] = useState<readonly PackListingRow[] | null>(null);
  const [failure, setFailure] = useState<ApiFailure | null>(null);
  const [importing, setImporting] = useState(false);
  const [lineages, setLineages] = useState<Readonly<Record<string, readonly SlotLineage[]>>>({});

  const reload = useCallback(async () => {
    const result = await fetchPacks();
    if (result.ok) {
      setPacks(result.value.packs);
      setFailure(null);
    } else {
      setFailure(result.failure);
    }
  }, []);

  useEffect(() => {
    void reload();
  }, [reload]);

  const showLineage = useCallback(async (pack: PackListingRow) => {
    const result = await fetchPackLineage(pack.pack_id, pack.version);
    if (result.ok) {
      setLineages((current) => ({ ...current, [pack.pack_id]: result.value.slots }));
    } else {
      setFailure(result.failure);
    }
  }, []);

  const exportPack = useCallback((pack: PackListingRow) => {
    // A download, not a fetch: the export is served as an attachment because it is a document
    // an operator keeps - the durable, portable form of a pack.
    window.location.assign(packExportUrl(pack.pack_id, pack.version));
  }, []);

  return (
    <>
      <PageHead
        title="Domain packs"
        subtitle="A pack is everything miForge needs to know about a domain"
        actions={
          <>
            <Button onClick={() => setImporting((current) => !current)}>Import pack</Button>
            <Button variant="primary" onClick={() => navigation.go('onboard')}>
              New pack
            </Button>
          </>
        }
      />

      <NoBuiltInDomainBanner className="mb-4" />

      {importing ? (
        <ImportPackDialog onImported={() => void reload()} onClose={() => setImporting(false)} />
      ) : null}

      {failure === null ? null : (
        <FailureNotice failure={failure} surface="the pack registry" className="mb-4" />
      )}

      {packs === null ? (
        <Card data-testid="packs-loading">Reading the pack registry…</Card>
      ) : packs.length === 0 ? (
        <Card data-testid="packs-empty">
          No pack is registered. Author one with <strong>New pack</strong>, which walks the seven
          steps and saves as it goes, or import an exported pack document. This installation ships
          no example packs — seeding unmarked ones that look real would be worse than an empty
          registry.
        </Card>
      ) : (
        <ul className="stack-y-3" data-testid="packs-list">
          {packs.map((pack) => (
            <PackCard
              key={`${pack.pack_id}@${pack.version}`}
              pack={pack}
              {...(lineages[pack.pack_id] === undefined ? {} : { lineage: lineages[pack.pack_id] })}
              onShowLineage={(target) => void showLineage(target)}
              onExport={exportPack}
            />
          ))}
        </ul>
      )}

      <PartIndex />
    </>
  );
}
