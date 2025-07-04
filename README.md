# CAN Protocol

Controll Area Network(CAN)는 1983년 Bosch사에서 개발한 자동차 내부 ECU들 간의 통신 프로토콜이다. 
Boadcasting-based(모든 수신자들에게 동시 전달), message-oriented(메세지 전달 후 각 수신자들이 적절한 조작 시행)통신이며 복잡한 배선을 줄이고 효과적인 통신을 가능하게 하였다.

```
외부 통신
└─ IVI(In-Vehicle Infotainment)
    └─ Gateway
        └─ CAN
            ├─ Engine control ECU
            ├─ Door control ECU
            ├─ Airbag ECU
            ├─ ...
```
차량 내 통신 구조는 위와 같은 구조로 되어 있다.
- IVI는 운전자가 조작 가능한 라디오, 스피커, 네비게이션과 같은 요소
- ECU는 각 부품(엔진, 창문 등)에 할당된 소규모 제어 유닛
- CAN은 이런 ECU들 간의 총체적인 통신을 이루어지게 함
- 사용자 조작을 실제 내부적 신호로 변환하고 메세지 중계를 해주는 것이 게이트웨이
- 여기에 더해, 최근에는 Cloud나 어플리케이션을 통한 외부 통신 접근도 이루어짐


### **1. 구조**
![image](https://github.com/user-attachments/assets/813f3ad0-cd0c-4c91-8a7e-24a4077ace9d)

CAN bus는 위와 같이 high와 low두 배선에 여러 contoll unit을 병렬로 연결하여 작동시킨다.
신호를 보내면 0(dominant), 신호를 보내지 않으면 1(recessive)상태가 된다.

![image](https://github.com/user-attachments/assets/5f461527-e76d-49c2-acd7-bf72bb467d34)

위 그림과 같이, 다른 unit에서 dominant 신호가 오게 되면, 상태를 recessive로 바꿔 더 이상 수신하지 않는다.


### **2. CAN frame**
![image](https://github.com/user-attachments/assets/1a5f1a81-3c73-483c-917b-f11997115ff8)
CAN 프로토콜에서 송수신되는 실제 데이터 프레임은 위와 같이 생겼다.
- SOF(Start of Frame)
- ID: 메세지 ID. 우선도를 포함하고 있다.
- RTR: 데이터 전송 여부. 1이면 수신, 0이면 송신이다.
- IDE: ID가 표준(11bit), 확장(29bit)중 어떤 것인지 구분
- DLC: 데이터 길이(0~8 bytes)
- Data: 실제 데이터
- CRC: 에러 검출용.
> 데이터(10110011)와 CRC 생성용 다항식(1101)가 있으면, 데이터 뒤에 000을 붙여 나눗셈을 한 나머지(011)가 CRC가 된다.
> 
> 송신할 때는 데이터+CRC(10110011011)를 보내게 되는데, 수신측에서 이걸 다항식으로 나누었을 때 나머지가 0이 아니라면 데이터 오류로 판별한다.
- ACK: 수신 실패 확인용
- EOF(End of Frame)
- 버퍼 구역


### **3. 특징**
- Multi-master: CAN bus에 연결된 모든 노드가 통신을 시작할 수 있다. 장점이기도 하지만 보안 취약점의 원인 중 하나.
- Bus topology: 배선이 간단해진다.
> CAN과 같은 구조가 없다면, 5개의 unit이 동시 통신을 하기 위해 5C2개의 배선이 필요해진다.
- 충돌 회피: 상기한 매커니즘으로 메세지 간 충돌이 없다.
- 신뢰성: 메세지 오류를 높은 확률로 검출할 수 있다.
- 실시간성: 동시 통신 가능



# CAN 프로토콜의 취약점

CAN 프로토콜은 과거에 개발된 통신 방법일 뿐 아니라, 외부와의 통신을 상정하지 않은 프로토콜이라 많은 보안상의 문제점을 포함하고 있다. 대표적인 보안 취약점은 아래와 같다.

- Eavesdropping(sniffing): 네트워크에 접속만 하면 모든 통신 수신 가능
- Spoofing: 네트워크에 접속만 하면 모든 메세지 송신 가능
- DoS(Denial-of-Service): 무작위, 혹은 에러를 유발시킬 수 있는 메세지를 반복적으로 보내 Bus-off 유도
- Replay Attack: 정상 메세지 중 하나를 캡쳐해 나중에 동일하게 사용하여 시스템 검사를 우회

### Jeep Cherokee hacking

2015년에 시연된 Jeep Cherokee Hacking은 이런한 CAN 프로토콜의 취약점을 이용해 실제로 게어권을 탈취한 사건이다.

1. Uconnect 시스템
Jeep 차량에서 IVI 역할을 하는 Uconnect 시스템은, 항상 셀룰러 네트워크에 연결되어 있다. 이 점을 이용해 Uconnect의 모든 포트를 스캔해, 접근할 수 있는 포트(6667)를 발견하였다.

2. CAN 메세지 송신
이후 쉘 권한을 획득해 IVI 시스템에서 직접적으로 IVI - Gatway - CAN으로 통하는 메세지를 송신할 수 있었다.

### 대응책?

- HSM(Hardware Secure Module)
- Autentication: cryptgographic technique. MAC(Message authentication Code)
- Encryption: 오버헤드가 적은 메세지 암호화
- Gateway firewall: 게이트웨이에서 IVI와 CAN을 분리시킴
- IDS(Industrial Detection Systems): 에러 검출 알고리즘, 머신러닝들...

가령, CAN에서 확장된 CAN FD라는 프레임도 제안되었는데, CAN에서 데이터 용량을 늘리고 ESI(Error State Indicator)를 추가한 등의 변경사항이 있다.

- 기존 프레임을 확장시켜 MAC 등을 도입할 수 있음
- ESI: CRC 에러가 반복해서 일어날 경우, ESI 비트로 신호를 보내 gateway에서 처리하게 함.



# 취약점이란?
- Weakness: Software error that can lead to vulnerability
- Vulnerability: Software weakness that can exploited by an attacker

https://www.cve.org/
https://cwe.mitre.org/

위 사이트에서 CVE(Common Weakness Enumeration)와 CWE(Common Vulnerabilities and Exposures)를 확인할 수 있다. 이런 보안상의 문제점은 Software Bug가 유발되는 것에서 기인한다.

 
### A well known ex1: Heartbleed

https://xkcd.com/1354/ <== 간단히 만화로 문제점을 표현한 것

```
memcpy(bp, pl, payload)
// payload >> pl이면?
```
OpenSSL에서 생긴 문제점으로, 웹서버에 패킷을 보낼 때, 실제 받아야하는 데이터 양보다 더 많은 데이터를 요구하여 생긴 문제. 이 남은 양의 데이터를 서버 뒷부분에서 채우면서, 타인의 쿼리 등 정보가 간단한 유출될 수 있음. 실제로 C에서 memcpy, strcpy 등의 함수를 권장하지 않고 있다.


### A well known ex2: goto fail;

```
if ((err = SSLHashSHA1.update(&hashCtx, &serverRandom)) != 0)
    goto fail;
if ((err = SSLHashSHA1.update(&hashCtx, &signedParams)) != 0)
    goto fail;
    goto fail; // <= Problematic
if ((err = SSLHashSHA1.final(&hashCtx, &hashOut)) != 0)
    goto fail;
```
MacOS와 iOS에서 생긴 보안 문제. SSL 인증서 검증 코드에서, goto fail; 이 두번 사용되면서, 모든 인증서가 허용됨.

```
fail:
    return err;
```
fail은 위와 같이 에러가 난 경우 에러 코드를 반환함.

```
// 예상 절차
SSLHashSHA1.update(...)
SSLHashSHA1.update(...)
SSLHashSHA1.final(...)
...

// 실제 절차
SSLHashSHA1.update(...)
SSLHashSHA1.update(...)
goto fail; // fail로 가지만 err = 0이 반환됨
```
위와 같이 모든 인증서가 해당 부분 이후로 검사되지 않아, 크나큰 보안 취약점이 되었음.


### A well known ex3: SQL Injection

ID - password 매칭으로 로그인 하는 웹사이트에서 DB조회를 위해 사용되는 SQL에 대한 공격

```
$mysqli->query("SELECT * FROM users WHERE username='{$username}' AND password='{$password}'");
```

위와 같이 로그인을 확인하는 코드가 있다고 할 때, password' OR 1=1 --을 비밀번호로 입력하게 되면,

```
SELECT * FROM users WHERE username='admin' and password='password' OR 1=1 --'
```

가 되어 항상 로그인에 성공하게 된다.


### Memory Safety: Buffer Overflow

버퍼에 할당된 메모리 양보다 더 큰 데이터를 읽거나 쓰는 행위. 예상되지 않는 오류를 발생시킨다.

![image](https://github.com/user-attachments/assets/902a9ceb-7259-48bf-b598-a61877879848)

메모리는 위와 같은 구조로 되어 있는데, 가장 기초적인 buffer overflow의 경우, Heap(동적으로 할당)이나 stack(정적으로 할당)에 많은 양의 데이터를 부하해 오류를 일으킨다. 둘은 약간의 차이가 있는데, stack-beased의 경우 인접 메모리를 덮어 써 원하는 코드를 실행시키는 것을 기대하고, heap-based의 경우 다른 heap 데이터, pointer 등을 덮어 써 의도치 않은 함수나 코드의 시행을 기대한다. 

scanf_s 등 API 수준에서의 확인으로 예방할 수도 있다.

---

- Canary: return address앞에 임의의 값을 넣고, 이 값이 변경되는 경우 문제가 있다고 판단하는 방어 방법
- Non-executable Memory(DEP): 특정 메모리 주소에서는 코드가 실행될 수 없게 하는 것.

---

**Return-to-libc**

Buffer overflow를 이용해 return address 등을 변경하여 DEP영역에 있는 syscall, libc등의 명령어를 시행하도록 하는 것. 

---

**ASRL(Address Space Layout Randomization)**

Stack, Heap, code 등의 base address를 무작위로 섞어 위의 방법으로 찾지 못하게 하는 것. 하지만 32-bit machine의 경우 brute-force로 주소를 찾을 수도 있음.

---

**ROP(Return Oriented Programming)**

```
1) ret → 0x08049010  → 실행: pop eax ; ret
   esp += 4 → eax ← 1

2) ret → 0x08049020  → 실행: pop ebx ; ret
   esp += 4 → ebx ← 2

3) ret → 0x08049030  → 실행: add eax, ebx
   eax = 3

4) ret → 0x08049040  → 실행: mov [ecx], eax
   → 원하는 주소에 3 기롣
```

위와 같이 ret가 반복되는 코드 부분(gadget)을 찾아, 이를 이용해 스택 너머로 이동 가능


# CTF? Wargame?

CTF(Capture the flag): 보안을 우회해 FLAG를 탈취하는 방식의 해킹 문제? 대회?
Wargame: 모의 해킹 환경을 구상해 원하는 목표를 달성하는 것. CTF도 이의 한 종류

---

**dreamhack wargame : session-basic**

```
#!/usr/bin/python3
from flask import Flask, request, render_template, make_response, redirect, url_for

app = Flask(__name__)

try:
    FLAG = open('./flag.txt', 'r').read()
except:
    FLAG = '[**FLAG**]'

users = {
    'guest': 'guest',
    'admin': FLAG
}

@app.route('/')
def index():
    username = request.cookies.get('username', None)
    if username:
        return render_template('index.html', text=f'Hello {username}, {"flag is " + FLAG if username == "admin" else "you are not admin"}')
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'GET':
        return render_template('login.html')
    elif request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        try:
            pw = users[username]
        except:
            return '<script>alert("not found user");history.go(-1);</script>'
        if pw == password:
            resp = make_response(redirect(url_for('index')) )
            resp.set_cookie('username', username)
            return resp 
        return '<script>alert("wrong password");history.go(-1);</script>'

app.run(host='0.0.0.0', port=8000)
```

../admin으로 가면

{"2576d97a2b84bace52e5f02cdcb3e5d7e694dfb6ce5b7b28409cf28d5b68d76e":"admin"}

을 확인할 수 있는데, 로그인 후 생긴 sessionid 쿠키를 해당 admin의 쿠키로 변경하면 admin으로 로그인 해 FLAG를 얻을 수 있다.

---

**webhacking.kr: old-01**

![image](https://github.com/user-attachments/assets/5eb4df0d-352c-4e84-a562-e34109483968)

```
<?php
  include "../../config.php";
  if($_GET['view-source'] == 1){ view_source(); }
  if(!$_COOKIE['user_lv']){
    SetCookie("user_lv","1",time()+86400*30,"/challenge/web-01/");
    echo("<meta http-equiv=refresh content=0>");
  }
?>
<html>
<head>
<title>Challenge 1</title>
</head>
<body bgcolor=black>
<center>
<br><br><br><br><br>
<font color=white>
---------------------<br>
<?php
  if(!is_numeric($_COOKIE['user_lv'])) $_COOKIE['user_lv']=1;
  if($_COOKIE['user_lv']>=4) $_COOKIE['user_lv']=1;
  if($_COOKIE['user_lv']>3) solve(1);
  echo "<br>level : {$_COOKIE['user_lv']}";
?>
<br>
<a href=./?view-source=1>view-source</a>
</body>
</html>
```

쿠키를 3보다 크고 4보다는 작은 실수로 바꾸면 해결할 수 있다.

이외에도 netcat을 이용한 시스템 해킹 문제도 시도 했는데, 해당 노트북을 집에 두고 와서 못 적었습니다...


---

**Dgitan Forensic**
2023 암호분석경진대회. PDF 참고.
