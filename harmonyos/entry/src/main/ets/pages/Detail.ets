import { router, promptAction } from '@kit.ArkUI';
import { Memo } from '../model/Memo';
import { DataManager } from '../utils/DataManager';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'MemoApp';

interface DetailArgs {
  memo: Memo;
}

@Entry
@Component
struct Detail {
  @State memo: Memo | null = null;
  @State draft: string = '';
  @State tag: string = '';

  async aboutToAppear() {
    try {
      const args = router.getParams() as DetailArgs;
      if (args && args.memo) {
        this.memo = args.memo;
        this.draft = args.memo.content;
        this.tag = args.memo.category;
      }
    } catch (e) {
      hilog.error(0x0000, TAG, 'get params failed %{public}s', JSON.stringify(e));
    }
  }

  async apply() {
    const current = this.memo;
    if (!current) {
      promptAction.openToast({ message: '数据异常，无法保存' });
      return;
    }
    try {
      const list = await DataManager.getMemos();
      const pos = list.findIndex(m => m.id === current.id);
      if (pos === -1) {
        return;
      }
      list[pos].content = this.draft;
      list[pos].category = this.tag;
      await DataManager.saveMemos(list);
      promptAction.openToast({ message: '修改已保存' });
      // 等数据落盘再返回，不然首页可能还是旧的
      setTimeout(() => {
        router.back();
      }, 200);
    } catch (e) {
      hilog.error(0x0000, TAG, 'save failed %{public}s', JSON.stringify(e));
    }
  }

  build() {
    Column({ space: 20 }) {
      Row() {
        Text('返回').fontSize(16).onClick(() => router.back())
        Text('备忘详情').fontSize(20).fontWeight(FontWeight.Bold).layoutWeight(1).textAlign(TextAlign.Center)
        Text('保存').fontSize(16).fontColor('#0A59F7').onClick(() => this.apply())
      }.width('100%').padding(15).backgroundColor(Color.White)

      if (this.memo) {
        TextArea({ text: this.draft })
          .height(150).width('90%')
          .onChange((v: string) => { this.draft = v; })

        Column({ space: 10 }) {
          Text('修改分类：').fontSize(16).fontWeight(FontWeight.Bold)
          Row({ space: 10 }) {
            ForEach(['工作', '生活', '学习'], (cat: string) => {
              Button(cat)
                .backgroundColor(this.tag === cat ? '#0A59F7' : '#CCCCCC')
                .onClick(() => { this.tag = cat; })
            })
          }
        }.width('90%').alignItems(HorizontalAlign.Start)

        Text('小贴士：如果 AI 预测错了，您可以在这里手动修改分类。')
          .fontSize(12).fontColor(Color.Gray).width('90%').margin({ top: 10 })
      } else {
        Text('加载中...').margin({ top: 50 })
      }
    }.width('100%').height('100%').backgroundColor('#F5F5F5')
  }
}
