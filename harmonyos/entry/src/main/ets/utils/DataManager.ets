import { preferences } from '@kit.ArkData';
import { Memo } from '../model/Memo';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'MemoApp';
const STORE_NAME: string = 'smart_memo';
const LIST_KEY: string = 'memo_list';

export class DataManager {
  private static store: preferences.Preferences | null = null;

  static async init(context: Context) {
    if (DataManager.store !== null) {
      return;
    }
    try {
      DataManager.store = await preferences.getPreferences(context, STORE_NAME);
      hilog.info(0x0000, TAG, 'store ready');
    } catch (e) {
      hilog.error(0x0000, TAG, 'open store failed %{public}s', JSON.stringify(e));
    }
  }

  static async getMemos(): Promise<Memo[]> {
    if (!DataManager.store) {
      return [];
    }
    try {
      const raw = await DataManager.store.get(LIST_KEY, '[]') as string;
      return JSON.parse(raw) as Memo[];
    } catch (e) {
      hilog.error(0x0000, TAG, 'read failed %{public}s', JSON.stringify(e));
      return [];
    }
  }

  static async saveMemos(list: Memo[]) {
    if (!DataManager.store) {
      return;
    }
    try {
      await DataManager.store.put(LIST_KEY, JSON.stringify(list));
      await DataManager.store.flush();
    } catch (e) {
      hilog.error(0x0000, TAG, 'write failed %{public}s', JSON.stringify(e));
    }
  }
}
