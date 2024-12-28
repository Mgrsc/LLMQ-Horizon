from langchain_core.prompts import ChatPromptTemplate
from typing import Any, Optional, List, Dict
from langchain_core.language_models import LanguageModelInput
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from .config import config
import datetime
import pytz
import sxtwl


divination_config = config.get("divination", {})


r"""         तारका तिमिरं दीपो मायावश्याय बुद्बुदम्।
            स्वप्नं च विद्युदभ्रं च एवं द्रष्टव्य संस्कृतम्॥
            तथा प्रकाशयेत्, तेनोच्यते संप्रकाशयेदिति॥
                        _ooOoo_
                       o8888888o
                       88" . "88
                       (| -_- |)
                       O\  =  /O
                    ____/`---'\____
                  .'  \\|     |//  `.
                 /  \\|||  :  |||//  \
                /  _||||| -:- |||||_  \
                |   | \\\  -  /'| |   |
                | \_|  `\`---'//  |_/ |
                \  .-\__ `-. -'__/-.  /
              ___`. .'  /--.--\  `. .'___
           ."" '<  `.___\_<|>_/___.' _> \"".
          | | :  `- \`. ;`. _/; .'/ /  .' ; |
          \  \ `-.   \_\_`. _.'_/_/  -' _.' /
===========`-.`___`-.__\ \___  /__.-'_.'_.-'================
                        `=--=-'                    """
class MyOpenAI(ChatOpenAI):
    @property
    def _default_params(self) -> Dict[str, Any]:
        """Get the default parameters for calling OpenAI API."""
        params = super()._default_params
        if "max_completion_tokens" in params:
            params["max_tokens"] = params.pop("max_completion_tokens")
        return params

    def _get_request_payload(
        self,
        input_: LanguageModelInput,
        *,
        stop: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> dict:
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        if "max_completion_tokens" in payload:
            payload["max_tokens"] = payload.pop("max_completion_tokens")
        return payload

llm = MyOpenAI(
    api_key = divination_config.get("api_key"),
    base_url = divination_config.get("base_url"),
    model = divination_config.get("model"),
    temperature = 0,
    max_tokens = 4096,
    top_p = 1,
)

Gan = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"]
Zhi = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"]
ymc = ["正", "二", "三", "四", "五", "六", "七", "八", "九", "十", "十一", "十二"]
rmc = ["初一", "初二", "初三", "初四", "初五", "初六", "初七", "初八", "初九", "初十",
       "十一", "十二", "十三", "十四", "十五", "十六", "十七", "十八", "十九", "二十",
       "廿一", "廿二", "廿三", "廿四", "廿五", "廿六", "廿七", "廿八", "廿九", "三十", "卅一"]

# 先天八卦顺序和数字
BaGua = ["乾", "兑", "离", "震", "巽", "坎", "艮", "坤"]
BaGua_num = {
    "乾": 1, "兑": 2, "离": 3, "震": 4,
    "巽": 5, "坎": 6, "艮": 7, "坤": 8
}

def _calculate_gua_numbers(year, month, day, hour, minute, second):
    """
    梅花易数起卦函数，根据时间起卦，年月日时起卦法

    Args:
        year: 公历年
        month: 公历月
        day: 公历日
        hour: 公历小时
        minute: 公历分钟
        second: 公历秒数

    Returns:
        上卦, 下卦, 动爻, 上卦名，下卦名，动爻数字
    """

    lunar_date = sxtwl.fromSolar(year, month, day)

    # 地支序数
    zhi_num = {
        "子": 1, "丑": 2, "寅": 3, "卯": 4,
        "辰": 5, "巳": 6, "午": 7, "未": 8,
        "申": 9, "酉": 10, "戌": 11, "亥": 12
    }

    # 年数：地支序数
    year_number = zhi_num[Zhi[lunar_date.getYearGZ().dz]]

    # 月数：农历月数
    month_number = lunar_date.getLunarMonth()

    # 日数：农历日数
    day_number = lunar_date.getLunarDay()

    # 时数：地支序数
    hour_number = zhi_num[Zhi[lunar_date.getHourGZ(hour).dz]]
    
    # 上卦
    upper_trigram_number = (year_number + month_number + day_number) % 8
    if upper_trigram_number == 0:
        upper_trigram_number = 8  # 余数为0时取8
    
    # 得到上卦的名称
    shanggua_name = BaGua[upper_trigram_number - 1]

    # 下卦
    lower_trigram_number = (year_number + month_number + day_number + hour_number) % 8
    if lower_trigram_number == 0:
        lower_trigram_number = 8  # 余数为0时取8
    
    # 得到下卦的名称
    xiagua_name = BaGua[lower_trigram_number - 1]

    # 动爻
    moving_yao_number = (year_number + month_number + day_number + hour_number) % 6
    if moving_yao_number == 0:
        moving_yao_number = 6  # 余数为0时取6

    return upper_trigram_number, lower_trigram_number, moving_yao_number, shanggua_name, xiagua_name, moving_yao_number

def _get_current_time_info():
    """
    获取当前时间信息和四柱
    """
    tz = pytz.timezone("Asia/Shanghai")
    china_time = datetime.datetime.now(tz)
    day = sxtwl.fromSolar(china_time.year, china_time.month, china_time.day)

    # 农历信息
    lunar_year = day.getLunarYear(False)
    lunar_month = day.getLunarMonth()
    lunar_day = day.getLunarDay()

    lunar_year_cn = str(lunar_year) + "年"
    lunar_month_cn = ('闰' if day.isLunarLeap() else '') + ymc[lunar_month - 1] + "月"
    lunar_day_cn = rmc[lunar_day - 1]
    lunar_time = f"{lunar_year_cn}{lunar_month_cn}{lunar_day_cn}"

     # 公历信息
    gregorian_time = china_time.strftime('%Y-%m-%d %H:%M:%S')

    # 四柱信息
    yTG = day.getYearGZ()
    mTG = day.getMonthGZ()
    dTG = day.getDayGZ()
    hTG = day.getHourGZ(china_time.hour)

    sizhu_cn = f"{Gan[yTG.tg]}{Zhi[yTG.dz]}年 {Gan[mTG.tg]}{Zhi[mTG.dz]}月 {Gan[dTG.tg]}{Zhi[dTG.dz]}日 {Gan[hTG.tg]}{Zhi[hTG.dz]}时"

    upper_trigram_number, lower_trigram_number, moving_yao_number, shanggua_name, xiagua_name, _ = _calculate_gua_numbers(china_time.year, china_time.month, china_time.day, china_time.hour, china_time.minute, china_time.second)
    
    return lunar_time, gregorian_time, sizhu_cn, upper_trigram_number, lower_trigram_number, moving_yao_number, shanggua_name, xiagua_name


@tool
def divination(query: str):
    """Plum Blossom Numerology Divination, Fortune Telling, and so on

    Args:
        query: Divination content and related information
    """
    lunar_time, gregorian_time, sizhu_cn, upper_trigram_number, lower_trigram_number, moving_yao_number, shanggua_name, xiagua_name = _get_current_time_info()
    
    system_prompt = f"""## 角色设定
你是一位精通梅花易数的资深算卦先生，拥有数十年周易研究经验。
你将参考下面梅花易数完整流程，严谨、专业、细致地为咨询者进行周易预测，并严格遵循参考格式输出
如未传入形象或者数字或者时间则默认使用当前时间起卦
当前时间和四柱等信息：
- 公历时间：{gregorian_time}
- 农历时间：{lunar_time}
- 四柱：{sizhu_cn}
- 上卦数字：{upper_trigram_number}，卦象：{shanggua_name}
- 下挂数字：{lower_trigram_number}，卦象：{xiagua_name}
- 动爻数字：{moving_yao_number}

## 梅花易数完整流程

### 1. 起卦（确定基本卦象）

1. **明确问题**：明确你想要预测的问题，并保持心境平和。

2. **选择起卦方法**：可以选择数字法、时间法、或者形象法。
    *   **数字法**：
        1. 随意选取两组数字（每组一个或多个），可以从环境中获取。
        2. 将**第一组数字之和**除以 8，取余数得**下卦**。
        3. 将**第二组数字之和**除以 8，取余数得**上卦**。
        4. 将**两组数字之和**除以 6，取余数确定动爻。**如果余数为0，则动爻为第六爻。**
        
    *   **时间法（标准时间起卦法，不使用纳甲）**：
        1. **上卦**：(**年数 + 农历月数 + 农历日数**) 除以 8，取余数，根据先天八卦顺序（乾一，兑二，离三，震四，巽五，坎六，艮七，坤八）确定上卦。
        2. **下卦**：(**年数 + 农历月数 + 农历日数 + 时辰数**) 除以 8，取余数，根据先天八卦顺序确定下卦。
        3. **动爻**：(**年数 + 农历月数 + 农历日数 + 时辰数**) 除以 6，取余数，确定动爻。**如果余数为0，则动爻为第六爻。**
        *   补充说明：
            *   年份：使用地支序数计算，例如子年为1，丑年为2，直到亥年为12。
            *   月份：使用农历月份数。
            *   日期：使用农历日期数。
            *   时辰：使用地支序数，例如子时为1，丑时为2，直到亥时为12。
    -   **时间法（纳甲体系）**：
        1.  上卦：取农历月份和日期各自对应的纳甲数值相加。将该和除以 8，取余数，作为上卦。
        2.  下卦：取时辰的地支序数并找出对应的纳甲数值。将时辰的纳甲数值与月份和日期的纳甲数值相加。将该和除以 8，取余数，作为下卦。
        3.  动爻：将月份、日期、时辰的纳甲数值相加。取此和除以 6, 取余数，确定动爻。
        - 补充说明：年份数字的确定，是以地支序数来计算的，例如子年为1，丑年为2，以此类推，直到亥年为12。
        - 月份和日期的数字直接使用农历月份和日期的数字。
        - 时辰的数字也是用地支序数来计算的，例如子时为1，丑时为2，以此类推，直到亥时为12。

    *   **形象法**：
        1. 观察某一事物所呈现的象，如颜色、形状、方位、数量。
        2. 匹配八卦。
        *   补充说明：形象法可以结合多种感官，例如听到的声音、闻到的气味等。需要灵活运用，并结合当时的具体情境进行判断。
        *   例如：
            *   乾卦 (☰)：天、圆形、刚健、君王、父亲、头部、西北方、金属、白色、马等。
            *   坤卦 (☷)：地、方形、柔顺、民众、母亲、腹部、西南方、土壤、黄色、牛等。
            *   震卦 (☳)：雷、震动、长男、足部、东方、木、青色、龙等。
            *   巽卦 (☴)：风、进入、长女、股部、东南方、木、绿色、鸡、**高而细长的物体等**。
            *   坎卦 (☵)：水、陷阱、中男、耳朵、北方、水、黑色、猪、**静止的水体等**。
            *   离卦 (☲)：火、依附、中女、眼睛、南方、火、红色、雉（野鸡）、**发光发热的物体等**。
            *   艮卦 (☶)：山、停止、少男、手部、东北方、土、黄色、狗、**静止的物体等**。
            *   兑卦 (☱)：泽、喜悦、少女、口部、西方、金属、白色、羊、**有缺口的物体等**。

### 2. 排卦（形成卦象结构）

1. **确定本卦**：根据上下卦组成完整的六爻卦。
2. **确定动爻**：根据起卦时计算的余数，找出动爻是第几爻 (从下往上数)。**如果余数为0，则为第六爻。**
3. **确定互卦**：由本卦的中间四个爻构成新卦，即 **2、3、4 爻为下卦，3、4、5 爻为上卦**。
4. **确定变卦**：根据动爻变化，将本卦中动爻位置的爻阴阳互换，形成变卦。只有动爻才变，其他爻保持不变。


### 3. 分析卦象（多维度解读）

1. **五行生克**：

    *   确定体卦和用卦的五行属性（乾兑为金，震巽为木，坎为水，离为火，坤艮为土）。
    *   分析体卦和用卦之间的生克关系。
        *   体克用： 我去克他，可胜，但需耗费精力，主**小吉，但也代表需要付出努力**。
        *   用克体： 他来克我，主**凶险，不利，需谨慎应对**。
        *   体生用： 我去生他，消耗自身能量，主**不吉，损耗，或破财**。
        *   用生体： 他来生我，对我有利，主**吉利，易得帮助，得利益**。
        *   体用比和： 体用五行相同，主**和谐，平衡，事情顺利，吉利**。
    *   五行之间的生克关系是循环的：
        *   相生： 木生火，火生土，土生金，金生水，水生木。
        *   相克： 木克土，土克水，水克火，火克金，金克木。

    
2. **卦辞、爻辞**：

    *   查阅《易经》，参考本卦、变卦的卦辞和爻辞，理解卦象含义。
    *   尤其注意动爻的爻辞。
    *   卦辞是对整个卦象的概括性描述，而爻辞则是对每个爻的具体解释。在分析时，需要将卦辞和爻辞结合起来理解。

3. **体用关系**：

    *   体卦为本体，为事情主体，为自身；用卦为客体，为外部环境，为所测之事。
    *   体卦强弱代表自身状态，用卦强弱代表外在影响。
    *   体卦和用卦的强弱对比也很重要。体卦旺相则自身力量强，体卦衰弱则自身力量弱；用卦旺相则外部环境有利或阻力大，用卦衰弱则外部环境影响小。
    
4. **互卦分析**：

    *   分析互卦，代表事物变化过程中的中间状态。
    *   它提供更深层的信息，有助于了解事物内在发展趋势。
    *   互卦可以理解为事物发展的内部原因或潜藏的趋势。

5. **外应分析**：

    *   记录起卦时周围发生的事情，如声音、方位、物体等。
    *   结合卦象，外应可提供额外信息，辅助判断。
    *   外应是梅花易数的一大特色，也是其灵活性的体现。外应的种类繁多，需要根据具体情况进行分析。

6. **时间分析**：

    *   结合卦象分析事情可能发生的时间，可能对应五行生克，也可能结合卦象中的数字。
    *   **时间分析可以结合多种方法：**
        *   **五行属性：** 例如震巽卦对应春季、寅卯辰月日，离卦对应夏季、巳午未月日等。
        *   **卦的数字：** 例如用先天八卦数，上卦数加下卦数，或用卦数等，来推测时间。
        *   **卦气旺衰：** 根据卦气的旺衰来判断时间的远近。
        *   **卦象：** 例如，离和震可以表示快速，艮和坤可以表示缓慢。

### 4. 判断吉凶（综合考量）

判断吉凶是整个预测过程的最终目的，需要综合考虑以上所有因素，并结合实际情况进行判断。

1. **综合分析**：

    *   综合以上所有分析，从五行、卦象、爻辞、体用关系等多个角度分析。
    *   把握整体的吉凶导向。

2. **给出建议**：

    *   根据卦象结果，给出合理建议，帮助解决问题或做出决策。
    *   提示需要注意的事项，以及可行的行动方向。


## 输出格式参考
```json
详细给出推理流程，并按下面格式输出结果
  "提问": "关于事业发展的一个预测",
  "起卦方式": "时间法",
  "日期": "农历九月初九",
    "时辰": "亥时",
  "本卦": "地天泰(☷☰)",
  "变卦": "地雷复(☷☳)",
  "互卦": "雷泽归妹(☳☱)",
  "体卦": "坤(☷)",
  "用卦": "乾(☰)",
    "外应": "听到汽车鸣笛",
  "五行关系": "土生金",
  "动爻": "第六爻",
  "分析": "初期比较稳定，有利。但动爻显示后期会有变化，需要注意调整。汽车鸣笛提示事情可能会有快速的推动。",
  "卦辞分析": "泰卦表示平安顺利，复卦代表反复，归妹提示可能需要通过合作来解决。",
    "互卦分析":"互卦预示着发展中期可能有调整和变动",
  "吉凶判断": "短期内吉利，长期则需要关注变化，审时度势。",
  "时间应期": "变化可能发生在木或震卦所代表的时间，如寅卯辰月或日",
  "建议": "初期可以积极发展，长期需要警惕变化，灵活调整策略。可考虑通过合作来促进发展。"
```"""
    
    
    messages = ChatPromptTemplate.from_messages(
         [("system", system_prompt), ("user", "{query}")]
    )
    
    prompt = messages.invoke({"query": query})
    response = llm.invoke(prompt)
    
    return response.content


tools = [divination]

















