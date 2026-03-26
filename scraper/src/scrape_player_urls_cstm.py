import csv
import asyncio
import os
from playwright.async_api import async_playwright

class SquadPlayerScraper:
    # Adicionei o parâmetro output_file aqui para facilitar a troca
    def __init__(self, squads_file="squads.txt", output_file="urls_squads_teste.csv"):
        self.squads_file = squads_file
        self.output_file = output_file
        self.all_player_urls = []
        self.seen_urls = set()

    async def scrape_squads(self):
        """Lê os links do squads.txt e extrai as URLs dos jogadores"""
        if not os.path.exists(self.squads_file):
            print(f"❌ Erro: Arquivo {self.squads_file} não encontrado!")
            return

        with open(self.squads_file, "r") as f:
            squad_urls = [line.strip() for line in f if line.strip() and not line.startswith("#")]

        if not squad_urls:
            print("⚠️ O arquivo squads.txt está vazio.")
            return

        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            )
            page = await context.new_page()

            # Bloqueia recursos para performance
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "stylesheet", "font", "media"] else route.continue_())

            for i, squad_url in enumerate(squad_urls):
                print(f"\n🚀 [{i+1}/{len(squad_urls)}] Acessando squad: {squad_url}")
                
                try:
                    await page.goto(squad_url, wait_until="domcontentloaded", timeout=30000)
                    await page.wait_for_selector("table tbody tr", timeout=10000)

                    urls = await page.evaluate("""
                        () => {
                            const anchors = document.querySelectorAll('table tbody tr td a[href*="/player/"]');
                            return [...new Set([...anchors].map(a => a.href))];
                        }
                    """)

                    new_count = 0
                    for url in urls:
                        clean_url = url.split('?')[0]
                        if clean_url not in self.seen_urls:
                            self.seen_urls.add(clean_url)
                            self.all_player_urls.append(clean_url)
                            new_count += 1
                    
                    print(f"  ✓ {new_count} novos jogadores adicionados.")
                    
                    # Salva no arquivo definido no __init__
                    self.save_urls_to_csv()

                except Exception as e:
                    print(f"  ✗ Erro em {squad_url}: {str(e)}")

            await browser.close()

    def save_urls_to_csv(self):
        """Salva a lista no arquivo definido em self.output_file"""
        with open(self.output_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['player_url'])
            for url in self.all_player_urls:
                writer.writerow([url])
        # Removi o print de backup para não poluir o terminal, 
        # mas o arquivo é atualizado a cada squad.

async def main():
    # ALTERE O NOME DO ARQUIVO AQUI ABAIXO:
    NOME_DO_ARQUIVO = "urls_squads_brasileirao.csv"
    
    scraper = SquadPlayerScraper(squads_file="squads.txt", output_file=NOME_DO_ARQUIVO)
    
    print("="*60)
    print(f"EXTRAÇÃO DE URLs -> ARQUIVO: {NOME_DO_ARQUIVO}")
    print("="*60)
    await scraper.scrape_squads()
    print(f"\n✨ Finalizado! Links salvos em: {NOME_DO_ARQUIVO}")

if __name__ == "__main__":
    asyncio.run(main())