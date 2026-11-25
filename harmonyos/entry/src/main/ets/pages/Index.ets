import { Memo } from '../model/Memo';
import { DataManager } from '../utils/DataManager';
import { router } from '@kit.ArkUI';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'MemoApp';

@Entry
@Component
struct Index {
  @State memos: Memo[] = [];
  @State keyword: string = '';
  private ready: boolean = false;

  async aboutToAppear() {
    try {
      await DataManager.init(getContext(this));
      this.ready = true;
      await this.reload();
    } catch (e) {
      hilog.error(0x0000, TAG, 'init failed %{public}s', JSON.stringify(e));
    }
  }

  onPageShow() {
    if (this.ready) {
      this.reload();
    }
  }

  async reload() {
    try {
      this.memos = await DataManager.getMemos();
    } catch (e) {
      hilog.error(0x0000, TAG, 'reload failed %{public}s', JSON.stringify(e));
    }
  }

  async removeMemo(id: string) {
    try {
      this.memos = this.memos.filter(m => m.id !== id);
      await DataManager.saveMemos(this.memos);
    } catch (e) {
      hilog.error(0x0000, TAG, 'remove failed %{public}s', JSON.stringify(e));
    }
  }

  async flipDone(item: Memo) {
    try {
      item.isDone = !item.isDone;
      await DataManager.saveMemos(this.memos);
      this.memos = this.memos.slice();
    } catch (e) {
      hilog.error(0x0000, TAG, 'flip failed %{public}s', JSON.stringify(e));
    }
  }

  shownMemos(): Memo[] {
    if (this.keyword.length === 0) {
      return this.memos;
    }
    const kw = this.keyword.toLowerCase();
    return this.memos.filter(m =>
    m.content.toLowerCase().includes(kw) || m.category.toLowerCase().includes(kw)
    );
  }

  build() {
    Column() {
      Text('我的智能备忘录')
        .fontSize(24)
        .fontWeight(FontWeight.Bold)
        .margin({ top: 20, bottom: 10 })
        .fontColor('#333333')

      Search({ value: this.keyword, placeholder: '搜索备忘内容或分类...' })
        .width('90%')
        .margin({ bottom: 10 })
        .onChange((value: string) => {
          this.keyword = value;
        })

      if (this.shownMemos().length === 0) {
        Column() {
          Text(this.keyword ? '🔍' : '📝').fontSize(50)
          Text(this.keyword ? '没有找到相关备忘' : '还没有备忘，快去添加一条吧！')
            .fontColor(Color.Gray)
            .margin({ top: 10 })
        }
        .layoutWeight(1)
        .justifyContent(FlexAlign.Center)
        .alignItems(HorizontalAlign.Center)
      } else {
        List({ space: 10 }) {
          ForEach(this.shownMemos(), (item: Memo) => {
            ListItem() {
              Row() {
                Toggle({ type: ToggleType.Checkbox, isOn: item.isDone })
                  .onChange(() => {
                    this.flipDone(item);
                  })
                  .margin({ right: 10 })

                Column() {
                  Text(item.content)
                    .fontSize(16)
                    .fontWeight(FontWeight.Medium)
                    .fontColor(item.isDone ? Color.Gray : '#1A1A1A')
                    .decoration({ type: item.isDone ? TextDecorationType.LineThrough : TextDecorationType.None })

                  Row() {
                    Text(item.category)
                      .fontSize(12)
                      .fontColor(Color.White)
                      .backgroundColor(item.isDone ? '#CCCCCC' : '#0A59F7')
                      .padding({ left: 6, right: 6, top: 2, bottom: 2 })
                      .borderRadius(4)
                    Text(item.createTime)
                      .fontSize(12)
                      .fontColor(Color.Gray)
                      .margin({ left: 10 })
                  }.margin({ top: 5 })
                }
                .alignItems(HorizontalAlign.Start)
                .layoutWeight(1)
                .onClick(() => {
                  router.pushUrl({
                    url: 'pages/Detail',
                    params: { memo: item }
                  });
                })

                Button('删除')
                  .fontSize(12)
                  .backgroundColor(Color.Red)
                  .onClick(() => {
                    this.removeMemo(item.id);
                  })
              }
              .width('100%')
              .padding(15)
              .backgroundColor(Color.White)
              .borderRadius(10)
            }
          }, (item: Memo) => item.id + item.category + item.content + item.isDone)
        }
        .width('100%')
        .layoutWeight(1)
        .padding({ left: 15, right: 15 })
      }

      Button('+ 新建备忘')
        .width('80%')
        .height(45)
        .margin({ bottom: 30, top: 10 })
        .backgroundColor('#0A59F7')
        .onClick(() => {
          router.pushUrl({ url: 'pages/AddMemo' });
        })
    }
    .width('100%')
    .height('100%')
    .backgroundColor('#F5F5F5')
  }
}
