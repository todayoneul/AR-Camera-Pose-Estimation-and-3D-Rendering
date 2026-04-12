# Media Branch Links (Git LFS)

대용량 영상은 `media-assets` 브랜치에서 Git LFS로 관리하고, 이 문서에서 경로와 링크를 정리합니다.

## 운영 규칙

- 100MB 초과 영상은 Git LFS로 추적
- main 브랜치에는 코드 + 미리보기 자산(GIF/JPG)만 유지
- 고화질 영상은 `media-assets` 브랜치에 커밋
- 파일 교체 시 브랜치 경로, 커밋 해시, 링크를 함께 갱신

## 브랜치/경로 템플릿

- 브랜치: `media-assets`
- 권장 경로:
	- `media/publicBench.mp4`
	- `media/cube.mp4`
	- `media/compare_test.mp4`
	- `media/final_test_noblock.mp4`

## 파일 매핑 표

| 파일 | 현재 로컬 경로 | 용량 | media-assets 브랜치 경로 | 커밋 해시 | 링크 |
|---|---|---:|---|---|---|
| Public Bench AR 원본 | publicBench.mp4 | 183MB | media/publicBench.mp4 | (입력 필요) | (입력 필요) |
| Cube AR 원본 | cube.mp4 | 120MB | media/cube.mp4 | (입력 필요) | (입력 필요) |
| 비교 영상 (원본\|AR) | output/compare_test.mp4 | 129MB | media/compare_test.mp4 | (입력 필요) | (입력 필요) |
| 안정화 테스트 영상 | output/final_test_noblock.mp4 | 131MB | media/final_test_noblock.mp4 | (입력 필요) | (입력 필요) |
| 입력 원본 영상 | IMG_0230.MOV | 98MB | media/IMG_0230.MOV | (선택) | (선택) |

## 업로드 커맨드 예시

~~~bash
git lfs install
git lfs track "*.mp4" "*.MOV" "*.mov"
git add .gitattributes
git commit -m "chore: enable lfs for media"

git switch -c media-assets
mkdir -p media
cp publicBench.mp4 media/
cp cube.mp4 media/
cp output/compare_test.mp4 media/
cp output/final_test_noblock.mp4 media/

git add media
git commit -m "media: add demo videos"
git push -u origin media-assets
git switch main
~~~

## 비고

- GitHub의 단일 파일 업로드 제한(100MB) 때문에, 브랜치 분리만으로는 해결되지 않습니다.
- 반드시 Git LFS를 함께 사용해야 안정적으로 push 가능합니다.
