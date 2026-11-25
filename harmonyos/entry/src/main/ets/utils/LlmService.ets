import { http } from '@kit.NetworkKit';
import { BusinessError } from '@kit.BasicServicesKit';
import { hilog } from '@kit.PerformanceAnalysisKit';

const TAG: string = 'MemoApp';
const ENDPOINT: string = 'https://open.bigmodel.cn/api/paas/v4/chat/completions';

export interface LlmResult {
  event: string;
  time: string;
  errorMsg?: string;
}

interface ChatMsg {
  role: string;
  content: string;
}

interface JsonFormat {
  type: string;
}

interface ChatBody {
  model: string;
  messages: ChatMsg[];
  response_format: JsonFormat;
}

interface ReplyMsg {
  content: string;
}

interface ReplyChoice {
  message: ReplyMsg;
}

interface ChatReply {
  choices: ReplyChoice[];
}

function stripCodeFence(raw: string): string {
  return raw.replace(/```json/g, '').replace(/```/g, '').trim();
}

export class LlmService {
  private static API_KEY: string = 'sk_xxxx';

  static async extractInfo(text: string): Promise<LlmResult> {
    const prompt = `从下面这段话里找出时间和事件，没写时间就填“未指定”。只返回 JSON：{"event": "事件", "time": "时间"}\n${text}`;

    const body: ChatBody = {
      model: 'glm-5.3-flash',
      messages: [{ role: 'user', content: prompt }],
      response_format: { type: 'json_object' }
    };

    const client = http.createHttp();
    try {
      const res = await client.request(ENDPOINT, {
        method: http.RequestMethod.POST,
        header: {
          'Content-Type': 'application/json',
          'Authorization': 'Bearer ' + LlmService.API_KEY
        },
        extraData: JSON.stringify(body)
      });

      if (res.responseCode !== 200) {
        hilog.error(0x0000, TAG, 'http %{public}d %{public}s', res.responseCode, String(res.result));
        return { event: text, time: '未指定', errorMsg: '服务返回 ' + res.responseCode };
      }

      const reply = JSON.parse(res.result as string) as ChatReply;
      const raw = reply.choices[0].message.content;
      hilog.info(0x0000, TAG, 'model raw %{public}s', raw);
      return JSON.parse(stripCodeFence(raw)) as LlmResult;
    } catch (err) {
      const code = (err as BusinessError).code;
      hilog.error(0x0000, TAG, 'request failed %{public}s', JSON.stringify(err));
      let tip = '网络请求失败，请稍后重试';
      if (code === 2300006 || code === 2300007 || code === 2300056 || code === 2300028) {
        tip = '网络连接超时，请检查您的网络设置';
      } else if (code === 2300060) {
        tip = '网络不可用，请确保设备已连接网络';
      }
      return { event: text, time: '未指定', errorMsg: tip };
    } finally {
      client.destroy();
    }
  }
}
