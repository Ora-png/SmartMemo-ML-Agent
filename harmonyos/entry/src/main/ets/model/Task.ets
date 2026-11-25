export interface Task {
  id: number;          // 必填字段1：唯一标识
  title: string;       // 必填字段2：任务标题
  isDone: boolean;     // 必填字段3：是否完成
  deadline?: string;   // 可选字段：截止日期（用?表示可选）
}