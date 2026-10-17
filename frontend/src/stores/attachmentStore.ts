/**
 * Which attachment the shell's resource strip shows (F11 task 16.4).
 *
 * Set by the run view from the open run's record and cleared when it closes, so the chip
 * appears only while an attached run is open - the strip is on every screen, and a chip
 * left behind from a run the operator navigated away from would describe nothing on screen.
 */
import { create } from 'zustand';
import type { AttachmentBlock } from '@/components/panels/hotupdate/weightsState';

interface AttachmentState {
  readonly attachment: AttachmentBlock | null;
  readonly show: (attachment: AttachmentBlock | null) => void;
}

export const useAttachmentStore = create<AttachmentState>((set) => ({
  attachment: null,
  show: (attachment) => set({ attachment }),
}));
