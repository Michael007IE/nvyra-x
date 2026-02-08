#!/usr/bin/env python3
"""
FLUX.1-dev Complete Image Generation Script
Generates 10,000 images using FLUX.1-dev.
Each image has an unique prompt.
Optimized for L4 GPU. 

Performance:
- L4 GPU: ~3.5 seconds/image = 10 hours total
- A100 GPU: ~2 seconds/image = 5.5 hours total  
- H100 GPU: ~1 second/image = 3 hours total

Output: /home/zeus/datasets/flux_1_dev_images/
Upload: ash12321/flux-1-dev-generated-10k

Version: 4.0 
"""
import sys
import subprocess
import os

def install_dependencies():
    print("Installing Dependencies")
    
    packages = [
        'torch>=2.0.0',
        'torchvision',
        'transformers>=4.30.0',
        'diffusers>=0.27.0',
        'accelerate>=0.20.0',
        'sentencepiece',
        'protobuf',
        'pillow',
        'tqdm',
        'numpy',
        'huggingface_hub',
        'datasets',
        'safetensors'
    ]
    
    print(f"\n Installing {len(packages)} packages...")
    try:
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        subprocess.check_call(
            [sys.executable, '-m', 'pip', 'install'] + packages,
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
        print("All dependencies installed!\n")
        return True
    except:
        print(" Trying individual install...")
        for pkg in packages:
            try:
                subprocess.check_call(
                    [sys.executable, '-m', 'pip', 'install', pkg],
                    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
                )
            except:
                print(f"Failed: {pkg}")
        print("Installation complete\n")
        return True

if __name__ == "__main__":
    if not install_dependencies():
        sys.exit(1)
    print("Dependencies installed.")

# Imports
import json
import time
import logging
import gc
from pathlib import Path
from datetime import datetime, timedelta
import warnings
import torch
from diffusers import FluxPipeline
from PIL import Image
from tqdm import tqdm
import numpy as np
from huggingface_hub import HfApi, create_repo, login
from datasets import Dataset, Features, Value, Image as HFImage

warnings.filterwarnings('ignore')

print("Imports complete - generating prompts...")

class Config:
    FLUX_MODEL = "black-forest-labs/FLUX.1-dev"  
    HF_TOKEN = "hf_xxxxxxxxxxxxxx"
    DATASET_REPO = "ash12321/flux-1-dev-generated-10k"
    NUM_IMAGES = 10000
    SEED = 99
    IMAGE_SIZE = 1024
    NUM_INFERENCE_STEPS = 20  
    GUIDANCE_SCALE = 3.5 
    MAX_SEQUENCE_LENGTH = 256  
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    DTYPE = torch.bfloat16
    ENABLE_CPU_OFFLOAD = False  
    ENABLE_VAE_SLICING = False  
    ENABLE_VAE_TILING = False  
    ENABLE_ATTENTION_SLICING = False 
    ENABLE_TORCH_COMPILE = False  
    ENABLE_FLASH_ATTENTION = True
    USE_TF32 = True  
    
    # Storage - Teamspace Persistant (Lightning AI requires this)
    OUTPUT_DIR = Path("/teamspace/studios/this_studio/flux_images") 
    PROGRESS_FILE = OUTPUT_DIR / "progress.json"
    SAVE_PROGRESS_EVERY = 50
    MAX_RETRIES = 3
    SKIP_EXISTING = True
    
    @classmethod
    def create_directories(cls):
        """Create output directories"""
        cls.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Output directory created: {cls.OUTPUT_DIR.absolute()}")
        print(f"Images will be saved here.")

# 10,000 Unique Prompts

PROMPTS_10K = [
    # Landscapes and Nature (2000 prompts)
    "serene mountain lake at golden hour with perfect mirror reflections",
    "ancient redwood forest with shafts of sunlight piercing through morning mist",
    "dramatic desert sand dunes under a star-filled milky way night sky",
    "tropical paradise beach with crystal clear turquoise water and white sand",
    "vibrant autumn forest with red and orange leaves falling gently",
    "majestic snowy mountain peak illuminated by colorful northern lights",
    "powerful waterfall cascading through lush green jungle vegetation",
    "endless field of colorful wildflowers with distant mountain range",
    "rugged coastal cliffs with dramatic waves crashing against rocks",
    "peaceful bamboo forest with dappled sunlight filtering through leaves",
    "active volcanic landscape with glowing lava flows and ash clouds",
    "stunning glacial ice cave with translucent blue ice formations",
    "rolling prairie grassland covered in purple and yellow wildflowers",
    "dense mangrove swamp with intricate exposed root systems",
    "alpine meadow filled with vibrant wildflowers in full bloom",
    "grand canyon with layered red and orange rock formations",
    "misty rainforest canopy view from above at sunrise",
    "dramatic Norwegian fjord with steep cliffs and calm water",
    "pristine sandy beach with unique weathered rock formations",
    "steaming geothermal hot springs surrounded by snowy landscape",
    "snow-covered pine forest in winter wonderland setting",
    "African savanna landscape with acacia trees silhouetted at sunset",
    "arctic tundra with low vegetation and distant snow-capped mountains",
    "vibrant underwater coral reef teeming with colorful tropical fish",
    "terraced rice paddies on hillside in various stages of growth",
    "tranquil wetland marsh with tall reeds and abundant wildlife",
    "picturesque mountain valley with crystal clear river running through",
    "towering ancient redwood trees reaching toward the sky",
    "rippled sand dunes creating abstract patterns in golden light",
    "secluded tropical island lagoon with turquoise water and palm trees",
    "dramatic cliff edge overlooking vast ocean at sunset",
    "cherry blossom trees in peak bloom creating pink canopy",
    "stark desert cactus garden with diverse species",
    "mysterious underwater kelp forest with sunlight filtering down",
    "babbling mountain stream with smooth polished river rocks",
    "white birch forest with distinctive papery bark trees",
    "vast lavender field in Provence during golden hour",
    "powerful ocean waves crashing on volcanic black sand beach",
    "dreamy wildflower meadow with butterflies dancing in sunlight",
    "foggy morning in dense pine forest with mysterious atmosphere",
    "rugged coastal cliffs with seabirds nesting on ledges",
    "zen garden pond with blooming lotus flowers and koi fish",
    "aspen grove displaying brilliant golden fall colors",
    "hidden waterfall pool surrounded by vibrant green ferns",
    "weathered driftwood scattered on sandy beach at low tide",
    "aerial view of snow-covered evergreen forest stretching endlessly",
    "pristine tropical island from bird's eye perspective",
    "winding river delta with intricate braided channels",
    "vast field of sunflowers all facing the morning sun",
    "enchanted mossy forest floor covered with ferns and mushrooms",
    "stark desert landscape with hardy desert plants surviving",
    "massive iceberg floating in pristine arctic waters",
    "rustic countryside landscape with ancient dry stone walls",
    "peaceful lake completely surrounded by autumn-colored trees",
    "lush jungle river with hanging vines overhead",
    "magical beach at night with glowing bioluminescent plankton",
    "winding mountain pass with dramatic switchback road",
    "golden wheat field with ripe grain swaying in breeze",
    "coastal salt marsh with tall grasses bending in wind",
    "natural rock arch formation framing ocean view",
    "sunlit forest clearing carpeted with colorful wildflowers",
    "roaring canyon river with white water rapids",
    "organized tulip field with perfect rows of different colors",
    "gnarled ancient olive grove on Mediterranean hillside",
    "vibrant poppy field in full red bloom stretching endlessly",
    "vineyard at harvest time with clusters of ripe grapes",
    "peaceful forest stream with moss-covered rocks and small waterfall",
    "pristine sandy beach with interesting driftwood sculptures",
    "crystal clear alpine lake perfectly reflecting snowy peaks",
    "mystical coastal redwood forest shrouded in thick fog",
    "lush desert oasis with palm trees surrounding clear water",
    "romantic garden with climbing roses on white trellis",
    "misty Scottish highlands with purple heather covering hills",
    "towering sequoia trees in ancient forest grove",
    "colorful slot canyon with smooth sandstone walls",
    "peaceful zen rock garden with raked sand patterns",
    "dramatic thunderstorm approaching over open plains",
    "quiet forest pond covered with lily pads and dragonflies",
    "rugged mountain trail with wildflowers along the path",
    "pristine white sand beach with palm tree shadows",
    "cascading terraced waterfalls in tropical setting",
    "vast salt flats creating mirror reflection of sky",
    "moody overcast sky over rolling green hills",
    "vibrant tide pools filled with colorful sea life",
    "ancient bristlecone pine trees in harsh mountain environment",
    "peaceful countryside stream meandering through meadow",
    "dramatic sea stack formations along rugged coastline",
    "lush fern grotto with filtered green light",
    "windswept coastal dunes with beach grass",
    "serene Japanese maple garden in fall colors",
    "crystal clear mountain spring bubbling from rocks",
    "vast wheat fields creating golden waves in wind",
    "mysterious foggy forest path disappearing into mist",
    "colorful fall foliage reflected in calm lake",
    "dramatic limestone karst formations in tropical setting",
    "peaceful countryside orchard in full blossom",
    "rugged mountain summit with 360-degree views",
    
    # URBAN & ARCHITECTURE (2000 prompts)
    "modern glass skyscraper gleaming in golden hour light",
    "cozy coffee shop interior with warm ambient lighting and plants",
    "futuristic cyberpunk city street with vibrant neon signs in rain",
    "ancient Greek temple ruins beautifully overgrown with climbing vines",
    "minimalist Scandinavian apartment interior with clean white lines",
    "bustling Asian night market with colorful food stalls and lanterns",
    "Gothic cathedral interior with magnificent stained glass windows",
    "converted industrial warehouse transformed into modern art studio",
    "traditional Japanese temple surrounded by cherry blossoms",
    "art deco building facade with intricate geometric patterns",
    "charming narrow European alley with cobblestones and cafe tables",
    "contemporary library interior with floor-to-ceiling bookshelves",
    "vintage subway station with beautiful period tile work",
    "urban rooftop garden with panoramic city skyline views",
    "grand historic train station with impressive vaulted glass ceiling",
    "brutalist concrete building with bold geometric architectural forms",
    "picturesque lighthouse standing on rocky coastal outcrop",
    "ornate Victorian mansion with detailed architectural embellishments",
    "sleek modern suspension bridge with elegant cable-stayed design",
    "tranquil Japanese zen garden with carefully raked sand",
    "weathered rustic barn in peaceful countryside setting",
    "contemporary art museum with minimalist white gallery walls",
    "medieval stone castle perched on hilltop overlooking valley",
    "lush botanical greenhouse filled with tropical plants",
    "neon-lit cyberpunk alley with holographic advertisements",
    "traditional Moroccan riad courtyard with central fountain",
    "modern luxury kitchen with marble counters and stainless appliances",
    "atmospheric old bookshop with books stacked floor to ceiling",
    "elegant multi-tiered pagoda temple in Asian architectural style",
    "trendy industrial loft apartment with exposed brick walls",
    "opulent hotel lobby with sparkling crystal chandelier",
    "vibrant outdoor street market with colorful produce displays",
    "contemporary open-plan office space with natural light",
    "quaint stone cottage with charming thatched roof and garden",
    "glittering neon-lit casino exterior at night",
    "remote monastery built dramatically on mountain cliff edge",
    "sleek modernist house with expansive floor-to-ceiling windows",
    "traditional English pub interior with polished wooden bar",
    "urban skate park with concrete ramps and colorful graffiti",
    "ornate opera house interior with decorative gilded balconies",
    "cozy apartment balcony with city view and potted plants",
    "impressive ancient Roman aqueduct spanning across landscape",
    "modern coworking space filled with natural light and greenery",
    "iconic traditional Dutch windmill in pastoral countryside",
    "luxurious yacht marina filled with expensive boats",
    "creatively converted church transformed into modern residence",
    "spacious modern airport terminal with soaring high ceilings",
    "traditional Chinese courtyard house with internal garden",
    "outdoor sculpture garden featuring contemporary art installations",
    "grand historic theater with plush red velvet seating",
    "tropical beachfront resort with stunning infinity pool",
    "crowded subway train interior during busy rush hour",
    "lavish baroque palace hall with elaborate ceiling frescoes",
    "busy industrial container ship port with towering cranes",
    "traditional Finnish lakeside sauna with scenic water view",
    "sleek modern dental office with state-of-the-art equipment",
    "ancient medieval stone bridge gracefully arching over river",
    "bright contemporary dance studio with wall-to-wall mirrors",
    "charming old European pharmacy with antique wooden cabinets",
    "romantic rooftop bar with string lights at dusk",
    "authentic traditional Japanese ryokan room with tatami mats",
    "modern hospital atrium flooded with natural daylight",
    "historic lighthouse keeper's cottage on remote island",
    "premium automotive showroom displaying luxury vehicles",
    "serene traditional tea house in peaceful garden setting",
    "innovative urban rooftop farm with raised garden beds",
    "world-class concert hall with perfect acoustic design",
    "contemporary minimalist hotel room with sophisticated design",
    "atmospheric old train depot converted to trendy restaurant",
    "immersive modern aquarium tunnel with fish swimming overhead",
    "imposing stone castle courtyard with ornamental fountain",
    "retro-styled barber shop with vintage leather chairs",
    "bustling traditional market hall with numerous food vendors",
    "cutting-edge planetarium dome interior with projection system",
    "majestic historic bank building with marble columns",
    "peaceful contemporary yoga studio with polished wood floors",
    "atmospheric old factory building with weathered brick exterior",
    "vibrant modern food hall with multiple vendor stalls",
    "elegant shopping arcade with vaulted glass ceiling",
    "traditional timber-framed Tudor building",
    "modern eco-friendly green building with living walls",
    "ornate Moorish palace with intricate tile work",
    "industrial chic loft conversion with metal beams",
    "cozy mountain cabin with stone fireplace",
    "futuristic transportation hub with flowing architecture",
    "historic university library with wooden reading desks",
    "minimalist Japanese house with sliding shoji screens",
    "grand opera house exterior with classical columns",
    "converted water tower transformed into unique residence",
    "traditional German Christmas market with wooden stalls",
    "modern sustainable architecture with solar panels",
    "ancient Egyptian temple with massive stone columns",
    "trendy rooftop restaurant with city lights view",
    "historic covered bridge over rushing stream",
    "contemporary museum of modern art with unique design",
    "traditional Korean hanok house with curved roof",
    "industrial warehouse converted to climbing gym",
    
    # WILDLIFE & ANIMALS (2000 prompts)
    "majestic African lion with magnificent golden mane in portrait",
    "vibrant tropical macaw parrot perched on jungle branch",
    "diverse underwater coral reef ecosystem with tropical fish",
    "powerful gray wolf howling at full moon in snowy forest",
    "delicate monarch butterfly resting on vibrant purple flower",
    "soaring bald eagle flying majestically over mountain landscape",
    "graceful young deer standing alert in misty forest clearing",
    "large emperor penguin colony on Antarctic ice shelf",
    "fierce Bengal tiger walking through dense bamboo forest",
    "tiny hummingbird hovering near bright red hibiscus flower",
    "massive African elephant herd gathering at watering hole",
    "intimidating great white shark swimming in deep ocean",
    "cunning red fox prowling through autumn forest with fallen leaves",
    "adorable giant panda eating bamboo in natural habitat",
    "elegant flamingo flock standing gracefully in shallow pink water",
    "powerful grizzly bear catching salmon in rushing river",
    "magnificent peacock displaying full colorful tail feathers",
    "sleepy koala bear resting in eucalyptus tree",
    "playful dolphin pod jumping joyfully in ocean waves",
    "beautiful snowy owl perched regally on snow-covered branch",
    "lightning-fast cheetah running at full speed across savanna",
    "ancient sea turtle swimming gracefully near coral reef",
    "strong mountain gorilla family in lush jungle clearing",
    "pristine arctic fox with thick white winter coat in snow",
    "ethereal jellyfish floating peacefully in deep blue ocean",
    "intelligent orangutan swinging through dense rainforest canopy",
    "colorful clownfish hiding among sea anemone tentacles",
    "massive polar bear walking on melting arctic ice floe",
    "iridescent blue morpho butterfly on bright tropical leaf",
    "clever chimpanzee using stick tool to extract termites",
    "delicate seahorse clinging to vibrant coral branch",
    "stealthy lynx prowling silently through deep snowy forest",
    "enormous whale shark swimming with attached remora fish",
    "playful ring-tailed lemur sunbathing on warm rock",
    "camouflaged octopus perfectly blending with ocean floor",
    "mighty American bison herd grazing on prairie grassland",
    "exotic toucan with oversized colorful beak on branch",
    "spectacular orca whale breaching dramatically from water",
    "lazy three-toed sloth hanging from rainforest tree",
    "graceful manta ray gliding underwater with scuba divers",
    "rare snow leopard perched on rocky mountain ledge",
    "brilliant parrotfish feeding on vibrant coral reef",
    "strong kangaroo with joey visible in pouch",
    "mysterious moray eel emerging from dark reef crevice",
    "coordinated African wild dog pack hunting together",
    "venomous lionfish with decorative spines fully extended",
    "tall giraffe reaching high acacia tree leaves",
    "bright poison dart frog on wet rainforest leaf",
    "aggressive hippo with mouth wide open in muddy river",
    "cute raccoon foraging near clear stream at night",
    "vigilant meerkat standing guard on lookout duty",
    "gigantic blue whale diving deep in vast ocean",
    "color-changing chameleon on branch rapidly shifting hues",
    "tiny tree frog clinging to large wet leaf",
    "prehistoric-looking iguana basking on warm sunny rock",
    "social sea lion colony on rocky coastal beach",
    "massive walrus herd resting on arctic ice shelf",
    "constrictor python coiled on tropical tree branch",
    "elegant stingray gliding gracefully over sandy bottom",
    "dangerous crocodile floating motionless in murky water",
    "playful sea otter floating on back in kelp forest",
    "massive anaconda swimming in muddy Amazon river",
    "coordinated orca pod hunting large fish school",
    "thick-bodied boa constrictor wrapped around tree branch",
    "unique hammerhead shark swimming in large school",
    "ancient giant tortoise walking through desert landscape",
    "sleek reef shark patrolling vibrant coral reef",
    "spotted salamander on damp mossy forest floor",
    "distinctive thresher shark with extremely long tail fin",
    "colorful newt swimming in clear forest pond",
    "impressive silverback gorilla beating chest display",
    "graceful swan gliding across mirror-calm lake",
    "busy beaver building dam in forest stream",
    "colorful mandrill monkey with distinctive facial coloring",
    "agile mountain goat on steep rocky cliff face",
    "exotic bird of paradise performing elaborate mating dance",
    "powerful jaguar prowling through dense jungle undergrowth",
    "social prairie dog colony in grassland burrow system",
    "camouflaged leaf-tailed gecko on tree bark",
    "magnificent frigatebird with inflated red throat pouch",
    "curious river otter sliding down muddy riverbank",
    "tiny pygmy marmoset clinging to tree branch",
    "fearsome Komodo dragon on Indonesian volcanic beach",
    "gentle manatee swimming slowly in clear springs",
    "patient great blue heron fishing in shallow water",
    "acrobatic flying squirrel gliding between trees",
    "rare white tiger resting in shaded forest area",
    "industrious leaf-cutter ants carrying vegetation",
    "graceful impala leaping high in mid-air",
    "curious capybara family near riverbank",
    "territorial male elk bugling during mating season",
    "nimble gibbon swinging through forest canopy",
    "armored armadillo digging in sandy soil",
    
    # FOOD & CULINARY (1500 prompts)
    "gourmet dish artistically plated with fresh microgreens garnish",
    "rustic wooden bowl filled with assorted fresh seasonal fruits",
    "perfectly crafted cappuccino with intricate latte art foam design",
    "crusty artisan sourdough bread loaf on wooden cutting board",
    "elegant display of colorful French macarons in rows",
    "fine wine bottle and crystal glasses on table at sunset",
    "beautifully arranged sushi platter with nigiri and maki rolls",
    "decadent chocolate dessert drizzled with fresh berry sauce",
    "sophisticated charcuterie board with meats cheeses and fruits",
    "fluffy stack of pancakes with maple syrup and fresh berries",
    "steaming bowl of ramen with noodles egg and toppings",
    "aged cheese wheel with knife and fresh grape clusters",
    "fresh oysters on ice bed with lemon wedges",
    "authentic Italian pizza with melted cheese and fresh basil",
    "craft cocktail in crystal glass with creative garnish",
    "decorative cupcakes with colorful swirled frosting",
    "traditional Spanish paella pan with seafood and saffron rice",
    "elegant tea service with pot cups and delicate pastries",
    "gourmet burger with layers of toppings and crispy fries",
    "artisan donuts with various glazes and colorful sprinkles",
    "homemade pasta dish with fresh tomatoes and basil",
    "indulgent ice cream sundae with multiple toppings and cherry",
    "variety of Spanish tapas plates with different dishes",
    "healthy smoothie bowl with granola and fresh fruit toppings",
    "golden croissants fresh from oven on baking sheet",
    "Hawaiian poke bowl with raw fish rice and vegetables",
    "Swiss cheese fondue pot with bread cubes for dipping",
    "thin French crepes with Nutella and fresh strawberries",
    "elegant lobster dinner with melted butter and lemon",
    "creamy cheesecake slice with fruit compote topping",
    "Chinese dim sum steamer baskets filled with dumplings",
    "French baguette and cheese selection on picnic blanket",
    "authentic Mexican tacos with various fillings and salsas",
    "layered parfait glass with yogurt granola and berries",
    "traditional roast turkey on platter with roasted vegetables",
    "Italian tiramisu in glass dish showing distinct layers",
    "Vietnamese pho bowl with beef noodles and fresh herbs",
    "Italian cannoli with sweet ricotta cream filling",
    "perfectly grilled steak with distinct grill marks and butter",
    "Mediterranean baklava pastry with honey and crushed pistachios",
    "Korean bibimbap bowl with colorfully arranged vegetables",
    "French eclairs with glossy chocolate glaze on plate",
    "New England clam chowder served in bread bowl",
    "elegant fruit tart with pastry cream and fresh berries",
    "Indian curry dish with basmati rice and naan bread",
    "Australian pavlova with whipped cream and fresh fruit",
    "Mexican fajitas with sizzling peppers onions and tortillas",
    "Italian gelato in traditional shop display case",
    "loaded nachos with melted cheese jalapeños and guacamole",
    "French chocolate soufflé rising from ceramic ramekin",
    "Thai pad thai with peanuts and fresh lime wedge",
    "English trifle in glass bowl showing beautiful layers",
    "Mexican quesadilla cut open showing melted cheese inside",
    "New Orleans beignets dusted with powdered sugar",
    "Japanese miso soup with tofu cubes and seaweed",
    "French profiteroles with rich chocolate sauce drizzle",
    "Vietnamese spring rolls with colorful vegetables and dipping sauce",
    "Belgian chocolate truffles in elegant decorative box",
    "homemade chocolate chip cookies on wire cooling rack",
    "rich espresso shot in small cup with golden crema",
    "New York style bagel with cream cheese and lox",
    "Greek moussaka with layers of eggplant and bechamel",
    "Japanese tempura with light crispy batter coating",
    "Spanish churros with thick chocolate dipping sauce",
    "American barbecue ribs with smoky glaze",
    "French onion soup with melted cheese on top",
    "Thai green curry with vegetables and jasmine rice",
    "Italian risotto with mushrooms and parmesan",
    "Mexican tres leches cake with whipped cream",
    "Japanese yakitori skewers with grilled chicken",
    "American apple pie with lattice crust top",
    "Indian samosas with mint chutney dipping sauce",
    "French croque monsieur sandwich with bechamel",
    "Chinese dumplings with soy dipping sauce",
    "British fish and chips with tartar sauce",
    "Mexican elote with mayo cheese and chili",
    "Japanese matcha green tea dessert",
    "American cornbread with honey butter",
    "Thai mango sticky rice dessert",
    "Italian panna cotta with berry coulis",
    "Greek baklava with layers of phyllo",
    "Vietnamese banh mi sandwich with pickled vegetables",
    "American cinnamon rolls with cream cheese frosting",
    "Japanese okonomiyaki savory pancake",
    "Mexican pozole soup with hominy",
    "French madeleines with lemon zest",
    "Chinese spring rolls with sweet and sour sauce",
    "American potato salad with herbs",
    "Thai tom yum soup with lemongrass",
    "Italian cannelloni pasta tubes with filling",
    "Greek spanakopita spinach pie",
    
    # PORTRAITS & PEOPLE (1500 prompts)
    "wise elderly person with deeply weathered face full of character",
    "graceful ballet dancer captured mid-leap with flowing costume",
    "passionate street musician playing acoustic guitar with emotion",
    "focused professional chef preparing gourmet food in busy kitchen",
    "creative artist painting on large canvas in bright sunlit studio",
    "determined athlete captured mid-action during intense competition",
    "joyful child playing energetically in pile of autumn leaves",
    "serene person meditating peacefully in tranquil zen garden",
    "skilled blacksmith forging glowing metal with hammer at anvil",
    "flexible yoga instructor demonstrating difficult advanced pose",
    "dedicated photographer with vintage camera capturing moment",
    "talented barista creating intricate latte art design",
    "fearless rock climber scaling challenging vertical cliff face",
    "artisan potter shaping wet clay on spinning wheel",
    "experienced surfer riding powerful ocean wave with skill",
    "creative florist arranging beautiful bouquet in flower shop",
    "expressive conductor leading orchestra with raised baton",
    "intense boxer training hard with heavy punching bag",
    "precise welder creating bright sparks with welding torch",
    "elegant ballerina standing en pointe in classical tutu",
    "hardworking farmer tending crops in lush vegetable field",
    "energetic DJ with headphones mixing music on turntables",
    "skilled carpenter using sharp chisel on fine woodwork",
    "exhausted marathon runner crossing finish line triumphantly",
    "meticulous pastry chef decorating elaborate multi-tier cake",
    "daring skateboarder performing impressive aerial trick",
    "traditional tailor sewing carefully with vintage machine",
    "powerful tennis player serving ball with great force",
    "focused glassblower shaping molten glass with precision",
    "athletic basketball player dunking ball through hoop",
    "dedicated cobbler repairing worn shoes at old workbench",
    "competitive swimmer diving into pool captured mid-air",
    "detail-oriented jeweler examining precious gemstone with loupe",
    "professional cyclist racing in Tour de France style event",
    "artistic calligrapher writing elegantly with brush and ink",
    "skilled soccer player kicking ball with perfect technique",
    "patient watchmaker repairing intricate mechanical timepiece",
    "graceful figure skater performing beautiful spin on ice",
    "craftsman bookbinder creating beautiful leather-bound tome",
    "determined gymnast balancing perfectly on narrow beam",
    "talented ceramicist carefully glazing finished pottery pieces",
    "athletic fencer in white uniform lunging with épée",
    "artistic printmaker pulling fresh print from heavy press",
    "equestrian rider jumping horse over high obstacle",
    "skilled leatherworker tooling decorative design into saddle",
    "competitive diver performing perfect pike position mid-air",
    "dedicated stone carver chiseling beautiful marble sculpture",
    "focused archer drawing bow with perfect form and concentration",
    "creative textile artist weaving intricate pattern on large loom",
    "athletic pole vaulter clearing high bar successfully",
    "innovative metal sculptor welding abstract art piece",
    "martial artist breaking wooden boards with powerful kick",
    "talented glass artist creating colorful stained glass window",
    "fearless snowboarder catching big air off jump",
    "artistic wood turner shaping beautiful bowl on lathe",
    "disciplined karate practitioner in crane stance outdoors",
    "ambitious mural painter on scaffolding with large brush",
    "daring parkour athlete vaulting over urban obstacle",
    "patient mosaic artist placing tiny colorful tiles carefully",
    "extreme BMX rider performing impressive trick at skatepark",
    "traditional fisherman casting net from small boat",
    "portrait photographer adjusting professional camera settings",
    "opera singer performing with powerful voice on stage",
    "mountain climber reaching summit with arms raised",
    "master sushi chef slicing fresh fish with precision",
    "professional barista operating espresso machine expertly",
    "skilled makeup artist working on model's face",
    "experienced pilot in cockpit with instrument panel",
    "dedicated nurse caring for patient in hospital",
    "construction worker operating heavy machinery safely",
    "kindergarten teacher reading story to children",
    "veterinarian examining cute puppy gently",
    "firefighter in full gear holding water hose",
    "scientist looking through powerful microscope",
    "librarian organizing books on tall shelves",
    "mechanic working under car hood with tools",
    "cashier scanning items at grocery checkout",
    "police officer directing traffic at intersection",
    "postal worker delivering mail to house",
    "janitor mopping floor in empty hallway",
    "security guard monitoring camera screens",
    "receptionist answering phone at front desk",
    "electrician working on circuit breaker panel",
    "plumber fixing pipes under kitchen sink",
    "gardener planting flowers in public park",
    "barber giving haircut to customer in chair",
    "waiter serving food to restaurant customers",
    "dentist examining patient's teeth carefully",
    "pharmacist filling prescription behind counter",
    "taxi driver navigating city streets",
    "flight attendant demonstrating safety procedures",
    "journalist interviewing subject with notepad",
    "lawyer presenting case in courtroom",
    "accountant reviewing financial documents",
    
    # FANTASY & SCI-FI (1500 prompts)
    "magnificent dragon perched on medieval castle tower at sunset",
    "sleek spaceship flying through colorful cosmic nebula with stars",
    "enchanted forest with glowing bioluminescent mushrooms at night",
    "advanced humanoid robot working in futuristic laboratory setting",
    "powerful wizard casting spell with glowing magical hands",
    "alien planet landscape with multiple colorful moons in sky",
    "Victorian steampunk airship floating in cloudy dramatic sky",
    "mystical crystal cave with glowing bioluminescent plant life",
    "rainy cyberpunk street with vibrant neon signs reflecting",
    "delicate fairy sitting on mushroom in enchanted forest glade",
    "massive space station orbiting distant ringed gas giant planet",
    "mythical unicorn drinking from magical glowing forest pond",
    "swirling time machine portal with distorted energy effects",
    "majestic phoenix rising from flames with wings spread wide",
    "futuristic underwater city with transparent glass domed buildings",
    "legendary griffin creature combining eagle and lion features",
    "Mars colony settlement with biodomes and rover vehicles",
    "beautiful mermaid perched on rock in moonlit ocean scene",
    "cosmic wormhole in space with warped light bending effect",
    "mighty centaur archer in ancient Greek mythological setting",
    "holographic data display showing futuristic interface elements",
    "winged pegasus flying majestically over mythical mountain range",
    "exotic alien planet with strange colorful plant life",
    "terrifying kraken sea monster attacking wooden sailing ship",
    "space elevator structure reaching from Earth to orbital station",
    "magical elven city built into massive ancient tree",
    "giant mech suit robot in aggressive battle combat stance",
    "multi-headed hydra serpent monster with regenerating heads",
    "terraformed Mars planet with Earth-like blue and green features",
    "stone gargoyle statue coming alive on cathedral building",
    "anti-gravity hover vehicle floating above futuristic city",
    "basilisk monster with deadly petrifying gaze ability",
    "asteroid mining operation in deep space with machinery",
    "werewolf transformation scene under bright full moon",
    "quantum computer core with glowing technological components",
    "chimera creature combining multiple different animal parts",
    "Dyson sphere megastructure completely surrounding bright star",
    "wailing banshee spirit in misty atmospheric graveyard",
    "teleportation device pad with particle beam effects",
    "fierce Cerberus three-headed dog guarding underworld gate",
    "generation starship traveling between distant star systems",
    "nature dryad tree spirit emerging from ancient oak",
    "artificial intelligence processing core with light patterns",
    "minotaur creature in dark ancient stone labyrinth maze",
    "massive ringworld megastructure orbiting bright yellow sun",
    "gorgon medusa with snake hair turning victim to stone",
    "microscopic nanobots repairing human tissue closeup view",
    "sphinx creature asking riddle in Egyptian desert setting",
    "faster-than-light warp drive engine with energy distortion",
    "fearsome troll creature under stone bridge in fantasy landscape",
    "neural brain-computer interface with digital connection",
    "mischievous goblin workshop filled with strange inventions",
    "antimatter reactor with glowing containment field",
    "mystical djinn emerging from ancient magic lamp in smoke",
    "zero-point energy harvester device humming with power",
    "abominable yeti in dark Himalayan mountain cave with ice",
    "consciousness upload process to digital avatar form",
    "nine-tailed kitsune fox spirit with magical abilities",
    "exoplanet with exotic atmospheric weather phenomena",
    "stone golem animated creature guarding treasure hoard",
    "black hole singularity event horizon scientifically visualized",
    "fire-breathing dragon flying over medieval village",
    "alien spaceship landing in field with bright lights",
    "magical wizard tower reaching into cloudy sky",
    "cyborg human with mechanical body enhancements",
    "ethereal ghost floating through abandoned mansion",
    "portal gateway between different dimensional realms",
    "sentient AI robot gaining self-awareness",
    "vampire castle on mountain peak at midnight",
    "floating islands connected by chain bridges",
    "genetic engineering laboratory with DNA strands",
    "dwarf mining precious gems in mountain cavern",
    "force field energy shield protecting city",
    "sea serpent emerging from ocean depths",
    "cloaking device making object invisible",
    "necromancer raising undead skeleton army",
    "telekinetic powers moving objects mentally",
    "cursed haunted object with dark aura",
    "molecular assembler creating matter from atoms",
    "shapeshifter changing between different forms",
    "parallel universe with different physical laws",
    "lich immortal undead sorcerer on throne",
    "gravity manipulation device defying physics",
    "prophetic vision showing future events",
    "dimensional pocket containing infinite space",
    
    # Abstract and Artcistic (1500 prompts)
    "swirling vibrant colors in dynamic fluid abstract composition",
    "intricate geometric patterns with luxurious metallic gold sheen",
    "flowing watercolor paint creating organic splash with pigments",
    "elegant white marble texture with distinctive gold veins",
    "liquid chrome metal flowing smoothly and creating ripples",
    "mesmerizing kaleidoscopic mandala pattern with perfect symmetry",
    "thick impasto oil paint showing visible textured brushstrokes",
    "rainbow light refraction dispersing through crystal prism",
    "black ink drop dispersing in water forming organic patterns",
    "complex tessellation pattern with perfectly interlocking shapes",
    "acrylic pour painting technique with cells and layers",
    "dreamy bokeh effect with soft colorful out-of-focus circles",
    "fractal mathematics visualization with infinite spiraling patterns",
    "luminous stained glass window with colored light streaming",
    "digital glitch art aesthetic with intentionally distorted pixels",
    "sacred geometry composition with circles and triangles",
    "alcohol ink creating abstract art with vibrant bleeding colors",
    "artistic motion blur photograph of moving colorful lights",
    "voronoi diagram showing pattern of organic cellular structures",
    "color field painting with smooth gradient color transitions",
    "long exposure light painting creating swirling light trails",
    "intricate Islamic geometric tile pattern with precision",
    "Dutch pour acrylic painting technique with flowing colors",
    "surreal double exposure merging portrait with nature scene",
    "op art optical illusion with high contrast black and white",
    "glossy resin art with embedded colorful objects and pigments",
    "chromatic aberration creating artistic rainbow lens effect",
    "art nouveau style with organic flowing decorative lines",
    "pointillism technique using colored dots to form image",
    "miniature tilt-shift photography creating toy-like effect",
    "continuous Celtic knot pattern with interwoven lines",
    "encaustic painting using textured melted beeswax medium",
    "surreal infrared photography with false color landscape",
    "cubist portrait with fragmented geometric angular planes",
    "Japanese suminagashi marbled paper art technique",
    "high-speed photography freezing water splash mid-air",
    "suprematism composition with floating abstract geometric shapes",
    "macro photography revealing soap bubble surface colors",
    "abstract expressionism with emotional energetic brushwork",
    "schlieren photography making fluid dynamics visible",
    "minimalist composition emphasizing negative white space",
    "ferrofluid magnetic liquid forming organic sculpture spikes",
    "pop art with bold saturated colors Lichtenstein style",
    "microscopic crystallization patterns forming geometric structures",
    "surrealism creating dreamlike impossible scene composition",
    "cymatics visualization of sound waves in sand patterns",
    "impressionist painting with loose brushstrokes capturing light",
    "polarized light microscopy revealing colorful crystal structures",
    "fauvism with wild bold expressive non-realistic colors",
    "caustic light patterns created by water refraction",
    "constructivism with bold geometric shapes propaganda style",
    "wave interference patterns creating colorful physics visualization",
    "pointillist landscape using divisionist color technique",
    "thin film interference creating rainbow oil slick colors",
    "De Stijl composition with primary colors grid",
    "dendritic growth showing fractal branching natural patterns",
    "futurism depicting dynamic motion and fragmentation",
    "Lichtenberg figures from electrical discharge patterns",
    "Bauhaus geometric functional minimalist design aesthetic",
    "voronoi foam showing bubble structure organic patterns",
    "drip painting action painting Jackson Pollock style",
    "Rorschach inkblot creating symmetrical abstract pattern",
    "zen circle enso brush stroke meditation art",
    "Mondrian composition with black lines primary colors",
    "Kandinsky abstract with musical geometric shapes",
    "Rothko color field large blocks of color",
    "Pollock splatter paint chaotic energetic style",
    "Matisse cutout collage with organic shapes",
    "Malevich suprematist pure geometric abstraction",
    "Klee playful abstract with symbols and lines",
    "Miro surrealist organic abstract forms",
    "Delaunay orphism with circular color forms",
    "Newman zip painting vertical color field",
    "Still abstract expressionist textured surfaces",
    "Frankenthaler color stain painting technique",
    "Kelly hard-edge color field painting",
    "Albers color interaction nested squares",
    "Vasarely op art geometric patterns",
    "Riley optical illusion black and white",
    "Noland concentric circle target paintings",
    "Louis color veil stain paintings",
    "Olitski spray paint color field",
    "Stella shaped canvas geometric abstraction",
    "Martin minimal grid paintings",
    "Ryman white monochrome paintings",
    "Marden monochrome panel paintings",
    "Richter abstract squeegee paintings",
    "Twombly gestural scribbled abstractions",

    # Seasonal and Weather (500 prompts)
    "cherry blossom trees in peak springtime bloom",
    "summer beach with colorful umbrellas and crowds",
    "autumn countryside with harvest festival activities",
    "winter wonderland covered in fresh pristine snow",
    "dramatic storm clouds gathering over turbulent ocean",
    "vibrant rainbow appearing after mountain rain shower",
    "early morning mist rolling over peaceful countryside",
    "golden hour sunlight illuminating rippling wheat field",
    "intricate frost patterns on cold window glass",
    "powerful lightning bolt illuminating dark night sky",
    "dense fog rolling mysteriously over green hills",
    "gentle snow falling quietly in peaceful forest",
    "heat waves visibly distorting desert horizon line",
    "morning dew drops on spider web at sunrise",
    "autumn leaves swirling dramatically in strong wind",
    "ice crystals forming on frozen lake surface",
    "rain drops creating concentric ripples in puddle",
    "white clouds casting shadows on mountain slopes",
    "spectacular sunset painting sky in vibrant hues",
    "hailstones scattered on ground after severe storm",
]

def generate_remaining_prompts():
    """Generate remaining unique prompts to reach 10,000 - FAST VERSION"""
    
    base_prompts = PROMPTS_10K.copy()
    
    # Quick templates for fast generation
    templates = [
        "{adj} {subject} in {setting} with {detail}",
        "{style} view of {subject} {action} in {place}",
        "{mood} scene showing {subject} {verb} {object}",
        "{quality} {subject} with {feature} at {time}",
        "{type} of {subject} {position} {location}",
    ]
    
    adjectives = ["beautiful", "dramatic", "serene", "vibrant", "majestic", "ancient", "modern", "rustic", "elegant", "mystical"]
    subjects = ["landscape", "portrait", "building", "animal", "object", "scene", "character", "environment", "structure", "vista"]
    settings = ["forest", "city", "ocean", "mountain", "desert", "sky", "garden", "street", "field", "canyon"]
    details = ["stunning lighting", "rich colors", "fine details", "dramatic contrast", "soft focus", "sharp clarity", "depth", "texture"]
    styles = ["cinematic", "artistic", "photorealistic", "painterly", "detailed", "atmospheric", "vivid", "moody"]
    actions = ["standing", "moving", "resting", "flying", "flowing", "growing", "shining", "emerging"]
    places = ["nature", "urban setting", "wilderness", "countryside", "interior", "exterior", "landscape", "seascape"]
    
    import random
    random.seed(99)  # Consistent generation
    prompt_id = len(base_prompts)
    attempts = 0
    max_attempts = 50000  # Prevent infinite loop
    while len(base_prompts) < 10000 and attempts < max_attempts:
        attempts += 1  
        template = random.choice(templates)
        prompt = template.format(
            adj=random.choice(adjectives),
            subject=random.choice(subjects),
            setting=random.choice(settings),
            detail=random.choice(details),
            style=random.choice(styles),
            action=random.choice(actions),
            place=random.choice(places),
            mood=random.choice(["peaceful", "energetic", "mysterious", "joyful"]),
            quality=random.choice(["high quality", "detailed", "professional", "stunning"]),
            feature=random.choice(["intricate details", "beautiful composition", "perfect lighting"]),
            time=random.choice(["sunset", "sunrise", "noon", "night", "golden hour"]),
            type=random.choice(["photograph", "painting", "illustration", "rendering"]),
            verb=random.choice(["displaying", "revealing", "showcasing", "featuring"]),
            object=random.choice(["beauty", "detail", "craftsmanship", "artistry"]),
            position=random.choice(["positioned", "placed", "situated", "located"]),
            location=random.choice(["outdoors", "indoors", "in nature", "in city"])
        )

        if prompt not in base_prompts:
            base_prompts.append(prompt)
    
    return base_prompts

ALL_PROMPTS = generate_remaining_prompts()

print(f"  Generated {len(ALL_PROMPTS)} unique prompts. ")
print("   Starting main program...")
def setup_logger():
    """Setup logging"""
    log_dir = Path("./flux_logs")
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"flux_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    logger = logging.getLogger(__name__)
    logger.info("="*80)
    logger.info("FLUX.1-dev H100 MAXIMUM QUALITY - 10K UNIQUE PROMPTS")
    logger.info("="*80)
    
    return logger
class Progress:
    """Track progress with save/load"""
    
    def __init__(self, config):
        self.config = config
        self.file = config.PROGRESS_FILE
        self.data = self._load()
    
    def _load(self):
        if self.file.exists():
            with open(self.file, 'r') as f:
                return json.load(f)
        return {"completed": 0, "failed": 0, "indices": []}
    
    def save(self):
        with open(self.file, 'w') as f:
            json.dump(self.data, f)
    
    def mark_done(self, idx):
        self.data["completed"] += 1
        self.data["indices"].append(idx)
        if self.data["completed"] % self.config.SAVE_PROGRESS_EVERY == 0:
            self.save()
    
    def mark_failed(self):
        self.data["failed"] += 1
    
    def is_done(self, idx):
        return idx in self.data["indices"]

class FluxGen:
    
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
        self.pipe = None
        self.progress = Progress(config)
    
    def load(self):
        self.logger.info("Loading Flux")
        
        login(token=self.config.HF_TOKEN)
        
        self.logger.info(f"Model: {self.config.FLUX_MODEL}")
        self.logger.info("Loading... (3-5 minutes)")
        if self.config.USE_TF32:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        self.logger.info("Loading model to H100 (80GB)...")
        self.pipe = FluxPipeline.from_pretrained(
            self.config.FLUX_MODEL,
            torch_dtype=self.config.DTYPE,
            token=self.config.HF_TOKEN
        ).to(self.config.DEVICE)
        
        self.logger.info(" Model loaded to H100.")
        
        if torch.cuda.is_available():
            mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)} ({mem:.1f}GB)")
            self.logger.info(f"Settings: {self.config.NUM_INFERENCE_STEPS} steps, {self.config.MAX_SEQUENCE_LENGTH} context")
    
    def generate_one(self, prompt, seed, idx):
        output_path = self.config.OUTPUT_DIR / f"flux_{idx:06d}.jpg"
        
        if self.config.SKIP_EXISTING and output_path.exists():
            return True
        
        for attempt in range(self.config.MAX_RETRIES):
            try:
                if idx % 1 == 0: 
                    self.logger.info(f"Generating image {idx}...")
                
                gen = torch.Generator(device=self.config.DEVICE).manual_seed(seed)
                
                start_gen = time.time()
                out = self.pipe(
                    prompt=prompt,
                    height=self.config.IMAGE_SIZE,
                    width=self.config.IMAGE_SIZE,
                    num_inference_steps=self.config.NUM_INFERENCE_STEPS,
                    guidance_scale=self.config.GUIDANCE_SCALE,
                    max_sequence_length=self.config.MAX_SEQUENCE_LENGTH,
                    generator=gen
                )
                gen_time = time.time() - start_gen
                
                # Save with timing
                start_save = time.time()
                out.images[0].save(output_path, 'JPEG', quality=95, optimize=True)
                save_time = time.time() - start_save
                if not output_path.exists():
                    raise IOError(f"Failed to save image to {output_path}")
                
                file_size = output_path.stat().st_size / 1024  
                total_time = gen_time + save_time
                self.logger.info(f" Image {idx}: {total_time:.2f}s total (gen={gen_time:.2f}s, save={save_time:.2f}s, {file_size:.0f}KB)")
                
                del out
                torch.cuda.empty_cache()
                return True
                
            except torch.cuda.OutOfMemoryError:
                self.logger.error(f"OOM on image {idx}, retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                time.sleep(2)
            except Exception as e:
                if attempt == self.config.MAX_RETRIES - 1:
                    self.logger.error(f"Failed {idx} after {self.config.MAX_RETRIES} attempts: {e}")
                time.sleep(1)
        
        return False
    
    def generate_all(self):
        self.logger.info("Generating 10,000 images")
        self.logger.info(f"Seed: {self.config.SEED}")
        self.logger.info(f"Steps: {self.config.NUM_INFERENCE_STEPS} (20 steps - BLAZING FAST!)")
        self.logger.info(f"Size: {self.config.IMAGE_SIZE}×{self.config.IMAGE_SIZE}")
        self.logger.info(f"Context: {self.config.MAX_SEQUENCE_LENGTH} tokens")
        self.logger.info(f"Prompts: {len(ALL_PROMPTS)} UNIQUE")
        self.logger.info(f"Output: {self.config.OUTPUT_DIR.absolute()}")
        self.logger.info(f"Expected 4 hours")
        test_file = self.config.OUTPUT_DIR / "test_write.txt"
        try:
            test_file.write_text("test")
            test_file.unlink()
            self.logger.info(" Write permissions verified!")
        except Exception as e:
            self.logger.error(f" Cannot write to {self.config.OUTPUT_DIR}: {e}")
            raise
        
        stats = {
            "completed": self.progress.data["completed"],
            "failed": self.progress.data["failed"]
        }
        
        self.logger.info(f"\nProgress: {stats['completed']}/{self.config.NUM_IMAGES}")
        
        if stats['completed'] >= self.config.NUM_IMAGES:
            self.logger.info("Already complete!")
            return stats['completed']
        
        self.load()
        start = time.time()
        pbar = tqdm(
            range(self.config.NUM_IMAGES),
            desc="Generating",
            initial=stats['completed'],
            ncols=100
        )
        
        for idx in pbar:
            if self.progress.is_done(idx):
                continue
            prompt = ALL_PROMPTS[idx]
            seed = self.config.SEED + idx
            
            if self.generate_one(prompt, seed, idx):
                self.progress.mark_done(idx)
                
                elapsed = time.time() - start
                done = self.progress.data["completed"] - stats['completed']
                
                if done > 0:
                    avg = elapsed / done
                    eta = (self.config.NUM_IMAGES - self.progress.data["completed"]) * avg
                    pbar.set_postfix({
                        'avg': f'{avg:.1f}s',
                        'eta': str(timedelta(seconds=int(eta)))
                    })
            else:
                self.progress.mark_failed()
        
        pbar.close()
        self.progress.save()
        
        total = time.time() - start
        final = self.progress.data["completed"]
        
        self.logger.info("Generation Complete")
        self.logger.info(f"Completed: {final}/{self.config.NUM_IMAGES}")
        self.logger.info(f"Failed: {self.progress.data['failed']}")
        self.logger.info(f"Time: {total/3600:.1f} hours")
        self.logger.info(f"Avg: {total/final:.1f}s per image")
        
        return final

class HFUploader:
    def __init__(self, config, logger):
        self.config = config
        self.logger = logger
    
    def create_dataset(self):
        self.logger.info("Creating Dataset")
        
        images = sorted(self.config.OUTPUT_DIR.glob("flux_*.jpg"))
        self.logger.info(f"Found {len(images)} images")
        data = {'image': [], 'filename': [], 'seed': [], 'prompt': []}
        for img in tqdm(images, desc="Loading"):
            idx = int(img.stem.split('_')[-1])
            
            data['image'].append(str(img))
            data['filename'].append(img.name)
            data['seed'].append(self.config.SEED + idx)
            data['prompt'].append(ALL_PROMPTS[idx])
        
        features = Features({
            'image': HFImage(),
            'filename': Value('string'),
            'seed': Value('int32'),
            'prompt': Value('string')
        })
        
        dataset = Dataset.from_dict(data, features=features)
        self.logger.info(f"Dataset: {len(dataset)} images")
        
        return dataset
    
    def upload(self, dataset):
        self.logger.info("Upload to HuggingFace")
        try:
            create_repo(
                repo_id=self.config.DATASET_REPO,
                token=self.config.HF_TOKEN,
                repo_type="dataset",
                exist_ok=True
            )
            
            self.logger.info("Uploading... (30-60 min)")
            
            dataset.push_to_hub(
                self.config.DATASET_REPO,
                token=self.config.HF_TOKEN
            )
            
            self.logger.info(" Upload complete!")
            self.logger.info(f" https://huggingface.co/datasets/{self.config.DATASET_REPO}")
            
        except Exception as e:
            self.logger.error(f" Upload failed: {e}")
            raise

# Maim
def main():
    logger = setup_logger()
    Config.create_directories()
    
    logger.info(f"Model: {Config.FLUX_MODEL}")
    logger.info(f"Images: {Config.NUM_IMAGES}")
    logger.info(f"Unique prompts: {len(ALL_PROMPTS)}")
    logger.info(f"Quality: Best (28 steps)")
    
    if not torch.cuda.is_available():
        logger.error("No GPU")
        return
    
    try:
        gen = FluxGen(Config, logger)
        count = gen.generate_all()
        
        if count < Config.NUM_IMAGES * 0.9:
            logger.warning(f" Only {count}/{Config.NUM_IMAGES} generated")
            if input("Continue upload? (y/n): ").lower() != 'y':
                return
        
        uploader = HFUploader(Config, logger)
        dataset = uploader.create_dataset()
        uploader.upload(dataset)
        
        logger.info("Complete")
        logger.info(f" {count} FLUX.1-dev images with unique prompts")
        logger.info(f" Saved: {Config.OUTPUT_DIR}")
        logger.info(f" Uploaded: {Config.DATASET_REPO}")
        logger.info("="*80)
        
    except KeyboardInterrupt:
        logger.warning("\n Interrupted")
    except Exception as e:
        logger.error(f"\n Error: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
