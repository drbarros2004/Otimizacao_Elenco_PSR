import argparse
import csv
import asyncio
import os
import random
from playwright.async_api import async_playwright
from player_scraper import PlayerScraper

# Importação blindada para o ambiente Python 3.13
try:
    import playwright_stealth
    if hasattr(playwright_stealth, "stealth") and callable(playwright_stealth.stealth):
        stealth_func = playwright_stealth.stealth
    else:
        from playwright_stealth.stealth import stealth as sf
        stealth_func = sf
except Exception:
    stealth_func = None

class SoFIFAScraper:
    def __init__(self, player_urls_file="player_urls.csv", output_file="player_stats.csv"):
        self.player_urls_file = player_urls_file
        self.output_file = output_file
        self.player_urls = []
        self.all_columns = [
            'player_id', 'version', 'name', 'full_name', 'description', 'image',
            'height_cm', 'weight_kg', 'dob', 'positions', 'overall_rating', 'potential',
            'value', 'wage', 'preferred_foot', 'weak_foot', 'skill_moves',
            'international_reputation', 'body_type', 'real_face',
            'release_clause', 'specialities', 'club_id', 'club_name', 'club_league_id',
            'club_league_name', 'club_logo', 'club_rating', 'club_position',
            'club_kit_number', 'club_joined', 'club_contract_valid_until',
            'country_id', 'country_name', 'country_league_id', 'country_league_name',
            'country_flag', 'country_rating', 'country_position', 'country_kit_number',
            'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy', 
            'attacking_short_passing', 'attacking_volleys',
            'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 
            'skill_ball_control',
            'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 
            'movement_reactions', 'movement_balance',
            'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 
            'power_long_shots',
            'mentality_aggression', 'mentality_interceptions', 'mentality_att_positioning', 
            'mentality_vision', 'mentality_penalties', 'mentality_composure',
            'defending_defensive_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
            'goalkeeping_gk_diving', 'goalkeeping_gk_handling', 'goalkeeping_gk_kicking', 
            'goalkeeping_gk_positioning', 'goalkeeping_gk_reflexes',
            'play_styles', 'url'
        ]

    def load_player_urls(self):
        """Filtra apenas quem ainda não está no CSV final"""
        if not os.path.exists(self.player_urls_file):
            print(f"❌ Erro: {self.player_urls_file} não encontrado.")
            return []

        with open(self.player_urls_file, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            next(reader) 
            all_urls = [row[0] for row in reader if row]
        
        scraped_urls = set()
        if os.path.exists(self.output_file):
            with open(self.output_file, 'r', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if row.get('url'): scraped_urls.add(row['url'])
            print(f"✅ Retomando: {len(scraped_urls)} já processados.")

        self.player_urls = [url for url in all_urls if url not in scraped_urls]
        return self.player_urls

    async def scrape_player_stats(self, max_players=None):
        async with async_playwright() as p:
            # Launcher mais limpo
            browser = await p.chromium.launch(headless=True)
            context = await browser.new_context(
                user_agent='Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
                locale='en-US'
            )
            page = await context.new_page()
            
            if stealth_func: stealth_func(page)
            await page.add_init_script("delete Object.getPrototypeOf(navigator).webdriver")

            # Bloqueio agressivo de lixo (anúncios, analytics, imagens)
            await page.route("**/*", lambda route: route.abort() if route.request.resource_type in ["image", "media", "font", "other"] or "google-analytics" in route.request.url or "doubleclick" in route.request.url else route.continue_())

            urls = self.player_urls[:max_players] if max_players else self.player_urls
            total = len(urls)

            for idx, url in enumerate(urls, 1):
                # Delay entre jogadores
                if idx > 1: await asyncio.sleep(random.uniform(2, 5))
                
                # Forçamos o idioma inglês
                url_with_lang = f"{url.rstrip('/')}/?hl=en-US"
                
                success = False
                retries = 0
                while retries < 2 and not success:
                    try:
                        print(f"[{idx}/{total}] Raspando: {url}")
                        
                        # MUDANÇA: wait_until="domcontentloaded" é muito mais rápido e estável
                        await page.goto(url_with_lang, wait_until="domcontentloaded", timeout=20000)
                        
                        # Espera apenas a tabela carregar (o que realmente importa)
                        await page.wait_for_selector('.grid', timeout=10000)
                        
                        # Pequeno scroll para garantir que o SoFIFA ative os scripts de dados
                        await page.mouse.wheel(0, 300)
                        await asyncio.sleep(1.5)

                        stats = await PlayerScraper.scrape_player_data(page, url)
                        
                        if stats and stats.get('name'):
                            # Se o rating falhar, espera um pouco mais e tenta de novo na mesma página
                            if not stats.get('overall_rating'):
                                await asyncio.sleep(2)
                                stats = await PlayerScraper.scrape_player_data(page, url)

                            self.save_player_to_csv(stats)
                            print(f"  ✓ {stats['name']} (Overall: {stats.get('overall_rating', 'N/A')})")
                            success = True
                        else:
                            retries += 1
                    except Exception as e:
                        print(f"  ✗ Erro (Tentativa {retries+1}): Timeout ou falha de carregamento.")
                        retries += 1
                        await asyncio.sleep(2)

            await browser.close()

    def save_player_to_csv(self, stats):
        file_exists = os.path.isfile(self.output_file)
        mode = 'a' if file_exists else 'w'
        with open(self.output_file, mode, newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.all_columns, extrasaction='ignore')
            if not file_exists:
                writer.writeheader()
            writer.writerow(stats)

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--player-urls-file", default="player_urls.csv")
    parser.add_argument("--output-file", default="player_stats.csv")
    parser.add_argument("--max-players", type=int, default=None)
    args = parser.parse_args()

    scraper = SoFIFAScraper(args.player_urls_file, args.output_file)
    scraper.load_player_urls()
    await scraper.scrape_player_stats(max_players=args.max_players)

if __name__ == "__main__":
    asyncio.run(main())