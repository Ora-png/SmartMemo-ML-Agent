import { Task } from '../model/Task';

@Component
export struct TaskItem {
  @Prop task: Task;
  onToggle: (id: number) => void = () => {};

  build() {
    Row() {
      this.statusIcon()

      Column() {
        Text(this.task.title)
          .fontSize(16)
          .fontWeight(this.task.isDone ? FontWeight.Normal : FontWeight.Medium)
          .fontColor(this.task.isDone ? $r('app.color.text_secondary') : $r('app.color.text_primary'))
          .decoration({ type: this.task.isDone ? TextDecorationType.LineThrough : TextDecorationType.None })

        if (this.task.deadline) {
          Text(`截止: ${this.task.deadline}`)
            .fontSize(12)
            .fontColor($r('app.color.danger'))
            .margin({ top: 4 })
        }
      }
      .alignItems(HorizontalAlign.Start)
      .layoutWeight(1)
      .margin({ left: 12 })

      Button(this.task.isDone ? '已完成' : '未完成')
        .fontSize(12)
        .fontColor(this.task.isDone ? $r('app.color.success') : $r('app.color.primary'))
        .backgroundColor(this.task.isDone ? $r('app.color.success_bg') : $r('app.color.primary_light'))
        .padding({ left: 10, right: 10, top: 4, bottom: 4 })
        .onClick(() => { this.onToggle(this.task.id); })
    }
    .width('100%')
    .padding(16)
    .borderRadius(12)
    .backgroundColor($r('app.color.card_bg'))
    .shadow({ radius: 8, color: '#1A000000', offsetX: 0, offsetY: 2 }) // 卡片阴影
    .margin({ bottom: 10 })
  }

  @Builder
  statusIcon() {
    Text(this.task.isDone ? '✓' : '○')
      .fontSize(20)
      .fontColor(this.task.isDone ? $r('app.color.success') : $r('app.color.text_secondary'))
      .width(32)
      .textAlign(TextAlign.Center)
  }
}