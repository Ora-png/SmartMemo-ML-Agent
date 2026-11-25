import { router, promptAction } from '@kit.ArkUI';
import { Memo } from '../model/Memo';
import { DataManager } from '../utils/DataManager';
import { MlEngine } from '../utils/MlEngine';
import { LlmService } from '../utils/LlmService';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'MemoApp';

@Entry
@Component
struct AddMemo {
  @State draft: string = '';
  @State eventText: string = '';
  @State timeText: string = '';
  @State category: string = '';
  @State isLoading: boolean = false;

  private classifier: MlEngine = new MlEngine();

  async aboutToAppear() {
    await this.classifier.initModel(getContext(this));
  }

  async parse() {
    if (this.draft.trim().length === 0) {
      promptAction.openToast({ message: '请输入内容' });
      return;
    }

    this.isLoading = true;
    this.eventText = '';
    this.timeText = '';
    this.category = '';

    try {
      const result = await LlmService.extractInfo(this.draft);
      this.eventText = result.event;
      this.timeText = result.time;
      this.category = this.classifier.predict(this.draft);

      if (result.errorMsg) {
        // 接口挂了就提示一下，分类继续用本地模型顶上
        promptAction.openToast({ message: result.errorMsg, duration: 5000 });
        hilog.info(0x0000, TAG, 'offline fallback, label=%{public}s', this.category);
        return;
      }

      hilog.info(0x0000, TAG, 'parsed, label=%{public}s', this.category);
      promptAction.openToast({ message: 'AI 解析成功' });
    } catch (e) {
      hilog.error(0x0000, TAG, 'parse error %{public}s', JSON.stringify(e));
    } finally {
      this.isLoading = false;
    }
  }

  async store() {
    if (this.draft.trim().length === 0) {
      promptAction.openToast({ message: '请输入内容' });
      return;
    }

    this.isLoading = true;
    try {
      const label = this.category.length > 0 ? this.category : '未分类';
      const body = this.eventText.length > 0 ? `[${this.timeText}] ${this.eventText}` : this.draft;

      const memo: Memo = {
        id: Date.now().toString(),
        content: body,
        category: label,
        isDone: false,
        createTime: new Date().toLocaleString()
      };

      const list = await DataManager.getMemos();
      list.push(memo);
      await DataManager.saveMemos(list);

      promptAction.openToast({ message: '保存成功' });
      router.back();
    } catch (e) {
      hilog.error(0x0000, TAG, 'save error %{public}s', JSON.stringify(e));
      promptAction.openToast({ message: '保存失败' });
    } finally {
      this.isLoading = false;
    }
  }

  build() {
    Column({ space: 15 }) {
      Row() {
        Text('取消').fontSize(16).onClick(() => router.back())
        Text('新建备忘录').fontSize(20).fontWeight(FontWeight.Bold).layoutWeight(1).textAlign(TextAlign.Center)
        Text('保存').fontSize(16).fontColor('#0A59F7').onClick(() => this.store())
      }.width('100%').padding(15).backgroundColor(Color.White)

      TextArea({ placeholder: '记点什么... 比如：明天下午三点去开项目碰头会' })
        .height(120).width('90%').margin({ top: 10 })
        .onChange((v: string) => { this.draft = v; })

      Button(this.isLoading ? 'Agent 解析中，请稍候...' : '🤖 运行 AI Agent 解析')
        .width('90%').backgroundColor('#0A59F7')
        .enabled(!this.isLoading)
        .onClick(() => { this.parse(); })

      if (this.eventText.length > 0) {
        Column({ space: 8 }) {
          Text('云端大模型提取结果：').fontSize(14).fontColor(Color.Gray)
          Text(`时间：${this.timeText}`).fontSize(16)
          Text(`事件：${this.eventText}`).fontSize(16)
          Row() {
            Text('本地 ML 预测分类：').fontSize(14)
            Text(this.category).fontSize(16).fontColor(Color.Red).fontWeight(FontWeight.Bold)
          }.margin({ top: 5 })
        }
        .width('90%').padding(15).backgroundColor('#FFF6E5').borderRadius(8).alignItems(HorizontalAlign.Start)
      }
    }
    .width('100%').height('100%').backgroundColor('#F5F5F5')
  }
}
