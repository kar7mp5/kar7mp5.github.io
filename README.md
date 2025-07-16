# kar7mp5.github.io

## Test for rendering

```bash
bundle exec jekyll serve --host 0.0.0.0 --port 4000 | ca
```

현재 Jekyll 서버가 정상적으로 실행되지 않고 있습니다.  
로그에 다음과 같은 에러가 반복적으로 나타납니다:

```
/usr/lib/ruby/3.0.0/socket.rb:201:in `bind': Address already in use - bind(2) for 0.0.0.0:4000 (Errno::EADDRINUSE)
```

이 에러는 **이미 4000번 포트에서 Jekyll 서버가 실행 중**이기 때문에 새로 서버를 띄울 수 없다는 의미입니다.

---

## 해결 방법

1. **기존 Jekyll 서버 프로세스 종료**
   - 터미널에서 아래 명령어를 입력하세요:
     ```sh
     lsof -i :4000
     ```
     또는
     ```sh
     netstat -tulpn | grep 4000
     ```
   - 결과에 나온 PID(프로세스 번호)를 확인하고, 아래 명령어로 종료하세요:
     ```sh
     kill -9 [PID]
     ```
   - 여러 개가 있으면 모두 종료합니다.

2. **서버 재실행**
   - 아래 명령어로 다시 서버를 실행하세요:
     ```sh
     bundle exec jekyll serve --host 0.0.0.0 --port 4000
     ```
   - 실행 후, 브라우저에서  
     `http://localhost:4000/2025/07/16/welcome-to-my-dev-blog.html`  
     등으로 접속하면 정상적으로 페이지가 나와야 합니다.

---

**정리**
- 4000번 포트에서 이미 실행 중인 서버를 모두 종료해야 새로 띄울 수 있습니다.
- 서버가 정상적으로 실행되면, 404가 아닌 실제 블로그 글이 보일 것입니다.

문제가 계속된다면,  
- `ps aux | grep jekyll` 결과  
- 또는 추가 에러 메시지  
를 알려주시면 더 도와드릴 수 있습니다!