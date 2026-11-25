import { util } from '@kit.ArkTS';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'MemoApp';

interface Weights {
  classes: string[];
  class_log_prior: number[];
  vocabulary: string[];
  log_probs: number[][];
}

export class MlEngine {
  private labels: string[] = [];
  private priors: number[] = [];
  private wordIndex: Map<string, number> = new Map();
  private logProbs: number[][] = [];

  async initModel(context: Context) {
    try {
      const bytes = await context.resourceManager.getRawFileContent('model.json');
      const text = util.TextDecoder.create('utf-8').decodeToString(bytes);
      const w = JSON.parse(text) as Weights;

      this.labels = w.classes;
      this.priors = w.class_log_prior;
      this.logProbs = w.log_probs;
      for (let i = 0; i < w.vocabulary.length; i++) {
        this.wordIndex.set(w.vocabulary[i], i);
      }
      hilog.info(0x0000, TAG, 'model loaded, %{public}d words', w.vocabulary.length);
    } catch (e) {
      hilog.error(0x0000, TAG, 'load model failed %{public}s', JSON.stringify(e));
    }
  }

  private split(text: string): string[] {
    const parts: string[] = text.split('');
    for (let i = 0; i < text.length - 1; i++) {
      parts.push(text.substring(i, i + 2));
    }
    return parts;
  }

  predict(text: string): string {
    const parts = this.split(text);
    const score = this.priors.slice();

    for (const p of parts) {
      const idx = this.wordIndex.get(p);
      if (idx === undefined) {
        continue;
      }
      for (let c = 0; c < this.labels.length; c++) {
        score[c] += this.logProbs[c][idx];
      }
    }

    let best = 0;
    for (let c = 1; c < score.length; c++) {
      if (score[c] > score[best]) {
        best = c;
      }
    }
    return this.labels[best];
  }
}
