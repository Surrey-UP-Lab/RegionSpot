IMAGENET_CLASSES = ["tench", "goldfish", "great white shark", "tiger shark", "hammerhead shark", "electric ray", "stingray", "rooster", "hen", "ostrich", "brambling", "goldfinch", "house finch", "junco", "indigo bunting", "American robin", "bulbul", "jay", "magpie", "chickadee", "American dipper", "kite (bird of prey)", "bald eagle", "vulture", "great grey owl", "fire salamander", "smooth newt", "newt", "spotted salamander", "axolotl", "American bullfrog", "tree frog", "tailed frog", "loggerhead sea turtle", "leatherback sea turtle", "mud turtle", "terrapin", "box turtle", "banded gecko", "green iguana", "Carolina anole", "desert grassland whiptail lizard", "agama", "frilled-necked lizard", "alligator lizard", "Gila monster", "European green lizard", "chameleon", "Komodo dragon", "Nile crocodile", "American alligator", "triceratops", "worm snake", "ring-necked snake", "eastern hog-nosed snake", "smooth green snake", "kingsnake", "garter snake", "water snake", "vine snake", "night snake", "boa constrictor", "African rock python", "Indian cobra", "green mamba", "sea snake", "Saharan horned viper", "eastern diamondback rattlesnake", "sidewinder rattlesnake", "trilobite", "harvestman", "scorpion", "yellow garden spider", "barn spider", "European garden spider", "southern black widow", "tarantula", "wolf spider", "tick", "centipede", "black grouse", "ptarmigan", "ruffed grouse", "prairie grouse", "peafowl", "quail", "partridge", "african grey parrot", "macaw", "sulphur-crested cockatoo", "lorikeet", "coucal", "bee eater", "hornbill", "hummingbird", "jacamar", "toucan", "duck", "red-breasted merganser", "goose", "black swan", "tusker", "echidna", "platypus", "wallaby", "koala", "wombat", "jellyfish", "sea anemone", "brain coral", "flatworm", "nematode", "conch", "snail", "slug", "sea slug", "chiton", "chambered nautilus", "Dungeness crab", "rock crab", "fiddler crab", "red king crab", "American lobster", "spiny lobster", "crayfish", "hermit crab", "isopod", "white stork", "black stork", "spoonbill", "flamingo", "little blue heron", "great egret", "bittern bird", "crane bird", "limpkin", "common gallinule", "American coot", "bustard", "ruddy turnstone", "dunlin", "common redshank", "dowitcher", "oystercatcher", "pelican", "king penguin", "albatross", "grey whale", "killer whale", "dugong", "sea lion", "Chihuahua", "Japanese Chin", "Maltese", "Pekingese", "Shih Tzu", "King Charles Spaniel", "Papillon", "toy terrier", "Rhodesian Ridgeback", "Afghan Hound", "Basset Hound", "Beagle", "Bloodhound", "Bluetick Coonhound", "Black and Tan Coonhound", "Treeing Walker Coonhound", "English foxhound", "Redbone Coonhound", "borzoi", "Irish Wolfhound", "Italian Greyhound", "Whippet", "Ibizan Hound", "Norwegian Elkhound", "Otterhound", "Saluki", "Scottish Deerhound", "Weimaraner", "Staffordshire Bull Terrier", "American Staffordshire Terrier", "Bedlington Terrier", "Border Terrier", "Kerry Blue Terrier", "Irish Terrier", "Norfolk Terrier", "Norwich Terrier", "Yorkshire Terrier", "Wire Fox Terrier", "Lakeland Terrier", "Sealyham Terrier", "Airedale Terrier", "Cairn Terrier", "Australian Terrier", "Dandie Dinmont Terrier", "Boston Terrier", "Miniature Schnauzer", "Giant Schnauzer", "Standard Schnauzer", "Scottish Terrier", "Tibetan Terrier", "Australian Silky Terrier", "Soft-coated Wheaten Terrier", "West Highland White Terrier", "Lhasa Apso", "Flat-Coated Retriever", "Curly-coated Retriever", "Golden Retriever", "Labrador Retriever", "Chesapeake Bay Retriever", "German Shorthaired Pointer", "Vizsla", "English Setter", "Irish Setter", "Gordon Setter", "Brittany dog", "Clumber Spaniel", "English Springer Spaniel", "Welsh Springer Spaniel", "Cocker Spaniel", "Sussex Spaniel", "Irish Water Spaniel", "Kuvasz", "Schipperke", "Groenendael dog", "Malinois", "Briard", "Australian Kelpie", "Komondor", "Old English Sheepdog", "Shetland Sheepdog", "collie", "Border Collie", "Bouvier des Flandres dog", "Rottweiler", "German Shepherd Dog", "Dobermann", "Miniature Pinscher", "Greater Swiss Mountain Dog", "Bernese Mountain Dog", "Appenzeller Sennenhund", "Entlebucher Sennenhund", "Boxer", "Bullmastiff", "Tibetan Mastiff", "French Bulldog", "Great Dane", "St. Bernard", "husky", "Alaskan Malamute", "Siberian Husky", "Dalmatian", "Affenpinscher", "Basenji", "pug", "Leonberger", "Newfoundland dog", "Great Pyrenees dog", "Samoyed", "Pomeranian", "Chow Chow", "Keeshond", "brussels griffon", "Pembroke Welsh Corgi", "Cardigan Welsh Corgi", "Toy Poodle", "Miniature Poodle", "Standard Poodle", "Mexican hairless dog (xoloitzcuintli)", "grey wolf", "Alaskan tundra wolf", "red wolf or maned wolf", "coyote", "dingo", "dhole", "African wild dog", "hyena", "red fox", "kit fox", "Arctic fox", "grey fox", "tabby cat", "tiger cat", "Persian cat", "Siamese cat", "Egyptian Mau", "cougar", "lynx", "leopard", "snow leopard", "jaguar", "lion", "tiger", "cheetah", "brown bear", "American black bear", "polar bear", "sloth bear", "mongoose", "meerkat", "tiger beetle", "ladybug", "ground beetle", "longhorn beetle", "leaf beetle", "dung beetle", "rhinoceros beetle", "weevil", "fly", "bee", "ant", "grasshopper", "cricket insect", "stick insect", "cockroach", "praying mantis", "cicada", "leafhopper", "lacewing", "dragonfly", "damselfly", "red admiral butterfly", "ringlet butterfly", "monarch butterfly", "small white butterfly", "sulphur butterfly", "gossamer-winged butterfly", "starfish", "sea urchin", "sea cucumber", "cottontail rabbit", "hare", "Angora rabbit", "hamster", "porcupine", "fox squirrel", "marmot", "beaver", "guinea pig", "common sorrel horse", "zebra", "pig", "wild boar", "warthog", "hippopotamus", "ox", "water buffalo", "bison", "ram (adult male sheep)", "bighorn sheep", "Alpine ibex", "hartebeest", "impala (antelope)", "gazelle", "arabian camel", "llama", "weasel", "mink", "European polecat", "black-footed ferret", "otter", "skunk", "badger", "armadillo", "three-toed sloth", "orangutan", "gorilla", "chimpanzee", "gibbon", "siamang", "guenon", "patas monkey", "baboon", "macaque", "langur", "black-and-white colobus", "proboscis monkey", "marmoset", "white-headed capuchin", "howler monkey", "titi monkey", "Geoffroy's spider monkey", "common squirrel monkey", "ring-tailed lemur", "indri", "Asian elephant", "African bush elephant", "red panda", "giant panda", "snoek fish", "eel", "silver salmon", "rock beauty fish", "clownfish", "sturgeon", "gar fish", "lionfish", "pufferfish", "abacus", "abaya", "academic gown", "accordion", "acoustic guitar", "aircraft carrier", "airliner", "airship", "altar", "ambulance", "amphibious vehicle", "analog clock", "apiary", "apron", "trash can", "assault rifle", "backpack", "bakery", "balance beam", "balloon", "ballpoint pen", "Band-Aid", "banjo", "baluster / handrail", "barbell", "barber chair", "barbershop", "barn", "barometer", "barrel", "wheelbarrow", "baseball", "basketball", "bassinet", "bassoon", "swimming cap", "bath towel", "bathtub", "station wagon", "lighthouse", "beaker", "military hat (bearskin or shako)", "beer bottle", "beer glass", "bell tower", "baby bib", "tandem bicycle", "bikini", "ring binder", "binoculars", "birdhouse", "boathouse", "bobsleigh", "bolo tie", "poke bonnet", "bookcase", "bookstore", "bottle cap", "hunting bow", "bow tie", "brass memorial plaque", "bra", "breakwater", "breastplate", "broom", "bucket", "buckle", "bulletproof vest", "high-speed train", "butcher shop", "taxicab", "cauldron", "candle", "cannon", "canoe", "can opener", "cardigan", "car mirror", "carousel", "tool kit", "cardboard box / carton", "car wheel", "automated teller machine", "cassette", "cassette player", "castle", "catamaran", "CD player", "cello", "mobile phone", "chain", "chain-link fence", "chain mail", "chainsaw", "storage chest", "chiffonier", "bell or wind chime", "china cabinet", "Christmas stocking", "church", "movie theater", "cleaver", "cliff dwelling", "cloak", "clogs", "cocktail shaker", "coffee mug", "coffeemaker", "spiral or coil", "combination lock", "computer keyboard", "candy store", "container ship", "convertible", "corkscrew", "cornet", "cowboy boot", "cowboy hat", "cradle", "construction crane", "crash helmet", "crate", "infant bed", "Crock Pot", "croquet ball", "crutch", "cuirass", "dam", "desk", "desktop computer", "rotary dial telephone", "diaper", "digital clock", "digital watch", "dining table", "dishcloth", "dishwasher", "disc brake", "dock", "dog sled", "dome", "doormat", "drilling rig", "drum", "drumstick", "dumbbell", "Dutch oven", "electric fan", "electric guitar", "electric locomotive", "entertainment center", "envelope", "espresso machine", "face powder", "feather boa", "filing cabinet", "fireboat", "fire truck", "fire screen", "flagpole", "flute", "folding chair", "football helmet", "forklift", "fountain", "fountain pen", "four-poster bed", "freight car", "French horn", "frying pan", "fur coat", "garbage truck", "gas mask or respirator", "gas pump", "goblet", "go-kart", "golf ball", "golf cart", "gondola", "gong", "gown", "grand piano", "greenhouse", "radiator grille", "grocery store", "guillotine", "hair clip", "hair spray", "half-track", "hammer", "hamper", "hair dryer", "hand-held computer", "handkerchief", "hard disk drive", "harmonica", "harp", "combine harvester", "hatchet", "holster", "home theater", "honeycomb", "hook", "hoop skirt", "gymnastic horizontal bar", "horse-drawn vehicle", "hourglass", "iPod", "clothes iron", "carved pumpkin", "jeans", "jeep", "T-shirt", "jigsaw puzzle", "rickshaw", "joystick", "kimono", "knee pad", "knot", "lab coat", "ladle", "lampshade", "laptop computer", "lawn mower", "lens cap", "letter opener", "library", "lifeboat", "lighter", "limousine", "ocean liner", "lipstick", "slip-on shoe", "lotion", "music speaker", "loupe magnifying glass", "sawmill", "magnetic compass", "messenger bag", "mailbox", "tights", "one-piece bathing suit", "manhole cover", "maraca", "marimba", "mask", "matchstick", "maypole", "maze", "measuring cup", "medicine cabinet", "megalith", "microphone", "microwave oven", "military uniform", "milk can", "minibus", "miniskirt", "minivan", "missile", "mitten", "mixing bowl", "mobile home", "ford model t", "modem", "monastery", "monitor", "moped", "mortar and pestle", "graduation cap", "mosque", "mosquito net", "vespa", "mountain bike", "tent", "computer mouse", "mousetrap", "moving van", "muzzle", "metal nail", "neck brace", "necklace", "baby pacifier", "notebook computer", "obelisk", "oboe", "ocarina", "odometer", "oil filter", "pipe organ", "oscilloscope", "overskirt", "bullock cart", "oxygen mask", "product packet / packaging", "paddle", "paddle wheel", "padlock", "paintbrush", "pajamas", "palace", "pan flute", "paper towel", "parachute", "parallel bars", "park bench", "parking meter", "railroad car", "patio", "payphone", "pedestal", "pencil case", "pencil sharpener", "perfume", "Petri dish", "photocopier", "plectrum", "Pickelhaube", "picket fence", "pickup truck", "pier", "piggy bank", "pill bottle", "pillow", "ping-pong ball", "pinwheel", "pirate ship", "drink pitcher", "block plane", "planetarium", "plastic bag", "plate rack", "farm plow", "plunger", "Polaroid camera", "pole", "police van", "poncho", "pool table", "soda bottle", "plant pot", "potter's wheel", "power drill", "prayer rug", "printer", "prison", "projectile", "projector", "hockey puck", "punching bag", "purse", "quill", "quilt", "race car", "racket", "radiator", "radio", "radio telescope", "rain barrel", "recreational vehicle", "fishing casting reel", "reflex camera", "refrigerator", "remote control", "restaurant", "revolver", "rifle", "rocking chair", "rotisserie", "eraser", "rugby ball", "ruler measuring stick", "sneaker", "safe", "safety pin", "salt shaker", "sandal", "sarong", "saxophone", "scabbard", "weighing scale", "school bus", "schooner", "scoreboard", "CRT monitor", "screw", "screwdriver", "seat belt", "sewing machine", "shield", "shoe store", "shoji screen / room divider", "shopping basket", "shopping cart", "shovel", "shower cap", "shower curtain", "ski", "balaclava ski mask", "sleeping bag", "slide rule", "sliding door", "slot machine", "snorkel", "snowmobile", "snowplow", "soap dispenser", "soccer ball", "sock", "solar thermal collector", "sombrero", "soup bowl", "keyboard space bar", "space heater", "space shuttle", "spatula", "motorboat", "spider web", "spindle", "sports car", "spotlight", "stage", "steam locomotive", "through arch bridge", "steel drum", "stethoscope", "scarf", "stone wall", "stopwatch", "stove", "strainer", "tram", "stretcher", "couch", "stupa", "submarine", "suit", "sundial", "sunglasses", "dark glasses", "sunscreen", "suspension bridge", "mop", "sweatshirt", "swim trunks / shorts", "swing", "electrical switch", "syringe", "table lamp", "tank", "tape player", "teapot", "teddy bear", "television", "tennis ball", "thatched roof", "front curtain", "thimble", "threshing machine", "throne", "tile roof", "toaster", "tobacco shop", "toilet seat", "torch", "totem pole", "tow truck", "toy store", "tractor", "semi-trailer truck", "tray", "trench coat", "tricycle", "trimaran", "tripod", "triumphal arch", "trolleybus", "trombone", "hot tub", "turnstile", "typewriter keyboard", "umbrella", "unicycle", "upright piano", "vacuum cleaner", "vase", "vaulted or arched ceiling", "velvet fabric", "vending machine", "vestment", "viaduct", "violin", "volleyball", "waffle iron", "wall clock", "wallet", "wardrobe", "military aircraft", "sink", "washing machine", "water bottle", "water jug", "water tower", "whiskey jug", "whistle", "hair wig", "window screen", "window shade", "Windsor tie", "wine bottle", "airplane wing", "wok", "wooden spoon", "wool", "split-rail fence", "shipwreck", "sailboat", "yurt", "website", "comic book", "crossword", "traffic or street sign", "traffic light", "dust jacket", "menu", "plate", "guacamole", "consomme", "hot pot", "trifle", "ice cream", "popsicle", "baguette", "bagel", "pretzel", "cheeseburger", "hot dog", "mashed potatoes", "cabbage", "broccoli", "cauliflower", "zucchini", "spaghetti squash", "acorn squash", "butternut squash", "cucumber", "artichoke", "bell pepper", "cardoon", "mushroom", "Granny Smith apple", "strawberry", "orange", "lemon", "fig", "pineapple", "banana", "jackfruit", "cherimoya (custard apple)", "pomegranate", "hay", "carbonara", "chocolate syrup", "dough", "meatloaf", "pizza", "pot pie", "burrito", "red wine", "espresso", "tea cup", "eggnog", "mountain", "bubble", "cliff", "coral reef", "geyser", "lakeshore", "promontory", "sandbar", "beach", "valley", "volcano", "baseball player", "bridegroom", "scuba diver", "rapeseed", "daisy", "yellow lady's slipper", "corn", "acorn", "rose hip", "horse chestnut seed", "coral fungus", "agaric", "gyromitra", "stinkhorn mushroom", "earth star fungus", "hen of the woods mushroom", "bolete", "corn cob", "toilet paper"]

IMAGENET_FOLDER_NAMES = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02699494', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02776631', 'n02777292', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02788148', 'n02790996', 'n02791124', 'n02791270', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02807133', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02825657', 'n02834397', 'n02835271', 'n02837789', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02871525', 'n02877765', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02894605', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02927161', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966193', 'n02966687', 'n02971356', 'n02974003', 'n02977058', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02988304', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000247', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03032252', 'n03041632', 'n03042490', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03065424', 'n03075370', 'n03085013', 'n03089624', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03160309', 'n03179701', 'n03180011', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03208938', 'n03216828', 'n03218198', 'n03220513', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03290653', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388043', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03457902', 'n03459775', 'n03461385', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03529860', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03637318', 'n03642806', 'n03649909', 'n03657121', 'n03658185', 'n03661043', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03697007', 'n03706229', 'n03709823', 'n03710193', 'n03710637', 'n03710721', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733281', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03871628', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03887697', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03899768', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03967562', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04008634', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04081281', 'n04086273', 'n04090263', 'n04099969', 'n04111531', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04152593', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04200800', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04239074', 'n04243546', 'n04251144', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04264628', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04296562', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04356056', 'n04357314', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04372370', 'n04376876', 'n04380533', 'n04389033', 'n04392985', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04443257', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04462240', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04493381', 'n04501370', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04523525', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04592741', 'n04596742', 'n04597913', 'n04599235', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06359193', 'n06596364', 'n06785654', 'n06794110', 'n06874185', 'n07248320', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07590611', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07836838', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07892512', 'n07920052', 'n07930864', 'n07932039', 'n09193705', 'n09229709', 'n09246464', 'n09256479', 'n09288635', 'n09332890', 'n09399592', 'n09421951', 'n09428293', 'n09468604', 'n09472597', 'n09835506', 'n10148035', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']

IMAGENETS_919_FOLDER_NAMES = ['n01440764', 'n01443537', 'n01484850', 'n01491361', 'n01494475', 'n01496331', 'n01498041', 'n01514668', 'n01514859', 'n01518878', 'n01530575', 'n01531178', 'n01532829', 'n01534433', 'n01537544', 'n01558993', 'n01560419', 'n01580077', 'n01582220', 'n01592084', 'n01601694', 'n01608432', 'n01614925', 'n01616318', 'n01622779', 'n01629819', 'n01630670', 'n01631663', 'n01632458', 'n01632777', 'n01641577', 'n01644373', 'n01644900', 'n01664065', 'n01665541', 'n01667114', 'n01667778', 'n01669191', 'n01675722', 'n01677366', 'n01682714', 'n01685808', 'n01687978', 'n01688243', 'n01689811', 'n01692333', 'n01693334', 'n01694178', 'n01695060', 'n01697457', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01729322', 'n01729977', 'n01734418', 'n01735189', 'n01737021', 'n01739381', 'n01740131', 'n01742172', 'n01744401', 'n01748264', 'n01749939', 'n01751748', 'n01753488', 'n01755581', 'n01756291', 'n01768244', 'n01770081', 'n01770393', 'n01773157', 'n01773549', 'n01773797', 'n01774384', 'n01774750', 'n01775062', 'n01776313', 'n01784675', 'n01795545', 'n01796340', 'n01797886', 'n01798484', 'n01806143', 'n01806567', 'n01807496', 'n01817953', 'n01818515', 'n01819313', 'n01820546', 'n01824575', 'n01828970', 'n01829413', 'n01833805', 'n01843065', 'n01843383', 'n01847000', 'n01855032', 'n01855672', 'n01860187', 'n01871265', 'n01872401', 'n01873310', 'n01877812', 'n01882714', 'n01883070', 'n01910747', 'n01914609', 'n01917289', 'n01924916', 'n01930112', 'n01943899', 'n01944390', 'n01945685', 'n01950731', 'n01955084', 'n01968897', 'n01978287', 'n01978455', 'n01980166', 'n01981276', 'n01983481', 'n01984695', 'n01985128', 'n01986214', 'n01990800', 'n02002556', 'n02002724', 'n02006656', 'n02007558', 'n02009229', 'n02009912', 'n02011460', 'n02012849', 'n02013706', 'n02017213', 'n02018207', 'n02018795', 'n02025239', 'n02027492', 'n02028035', 'n02033041', 'n02037110', 'n02051845', 'n02056570', 'n02058221', 'n02066245', 'n02071294', 'n02074367', 'n02077923', 'n02085620', 'n02085782', 'n02085936', 'n02086079', 'n02086240', 'n02086646', 'n02086910', 'n02087046', 'n02087394', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02088632', 'n02089078', 'n02089867', 'n02089973', 'n02090379', 'n02090622', 'n02090721', 'n02091032', 'n02091134', 'n02091244', 'n02091467', 'n02091635', 'n02091831', 'n02092002', 'n02092339', 'n02093256', 'n02093428', 'n02093647', 'n02093754', 'n02093859', 'n02093991', 'n02094114', 'n02094258', 'n02094433', 'n02095314', 'n02095570', 'n02095889', 'n02096051', 'n02096177', 'n02096294', 'n02096437', 'n02096585', 'n02097047', 'n02097130', 'n02097209', 'n02097298', 'n02097474', 'n02097658', 'n02098105', 'n02098286', 'n02098413', 'n02099267', 'n02099429', 'n02099601', 'n02099712', 'n02099849', 'n02100236', 'n02100583', 'n02100735', 'n02100877', 'n02101006', 'n02101388', 'n02101556', 'n02102040', 'n02102177', 'n02102318', 'n02102480', 'n02102973', 'n02104029', 'n02104365', 'n02105056', 'n02105162', 'n02105251', 'n02105412', 'n02105505', 'n02105641', 'n02105855', 'n02106030', 'n02106166', 'n02106382', 'n02106550', 'n02106662', 'n02107142', 'n02107312', 'n02107574', 'n02107683', 'n02107908', 'n02108000', 'n02108089', 'n02108422', 'n02108551', 'n02108915', 'n02109047', 'n02109525', 'n02109961', 'n02110063', 'n02110185', 'n02110341', 'n02110627', 'n02110806', 'n02110958', 'n02111129', 'n02111277', 'n02111500', 'n02111889', 'n02112018', 'n02112137', 'n02112350', 'n02112706', 'n02113023', 'n02113186', 'n02113624', 'n02113712', 'n02113799', 'n02113978', 'n02114367', 'n02114548', 'n02114712', 'n02114855', 'n02115641', 'n02115913', 'n02116738', 'n02117135', 'n02119022', 'n02119789', 'n02120079', 'n02120505', 'n02123045', 'n02123159', 'n02123394', 'n02123597', 'n02124075', 'n02125311', 'n02127052', 'n02128385', 'n02128757', 'n02128925', 'n02129165', 'n02129604', 'n02130308', 'n02132136', 'n02133161', 'n02134084', 'n02134418', 'n02137549', 'n02138441', 'n02165105', 'n02165456', 'n02167151', 'n02168699', 'n02169497', 'n02172182', 'n02174001', 'n02177972', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02229544', 'n02231487', 'n02233338', 'n02236044', 'n02256656', 'n02259212', 'n02264363', 'n02268443', 'n02268853', 'n02276258', 'n02277742', 'n02279972', 'n02280649', 'n02281406', 'n02281787', 'n02317335', 'n02319095', 'n02321529', 'n02325366', 'n02326432', 'n02328150', 'n02342885', 'n02346627', 'n02356798', 'n02361337', 'n02363005', 'n02364673', 'n02389026', 'n02391049', 'n02395406', 'n02396427', 'n02397096', 'n02398521', 'n02403003', 'n02408429', 'n02410509', 'n02412080', 'n02415577', 'n02417914', 'n02422106', 'n02422699', 'n02423022', 'n02437312', 'n02437616', 'n02441942', 'n02442845', 'n02443114', 'n02443484', 'n02444819', 'n02445715', 'n02447366', 'n02454379', 'n02457408', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02483708', 'n02484975', 'n02486261', 'n02486410', 'n02487347', 'n02488291', 'n02488702', 'n02489166', 'n02490219', 'n02492035', 'n02492660', 'n02493509', 'n02493793', 'n02494079', 'n02497673', 'n02500267', 'n02504013', 'n02504458', 'n02509815', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02606052', 'n02607072', 'n02640242', 'n02641379', 'n02643566', 'n02655020', 'n02666196', 'n02667093', 'n02669723', 'n02672831', 'n02676566', 'n02687172', 'n02690373', 'n02692877', 'n02701002', 'n02704792', 'n02708093', 'n02727426', 'n02730930', 'n02747177', 'n02749479', 'n02769748', 'n02782093', 'n02783161', 'n02786058', 'n02787622', 'n02790996', 'n02791124', 'n02793495', 'n02794156', 'n02795169', 'n02797295', 'n02799071', 'n02802426', 'n02804414', 'n02804610', 'n02808304', 'n02808440', 'n02814533', 'n02814860', 'n02815834', 'n02817516', 'n02823428', 'n02823750', 'n02834397', 'n02835271', 'n02840245', 'n02841315', 'n02843684', 'n02859443', 'n02860847', 'n02865351', 'n02869837', 'n02870880', 'n02879718', 'n02883205', 'n02892201', 'n02892767', 'n02895154', 'n02906734', 'n02909870', 'n02910353', 'n02916936', 'n02917067', 'n02930766', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02951585', 'n02963159', 'n02965783', 'n02966687', 'n02971356', 'n02978881', 'n02979186', 'n02980441', 'n02981792', 'n02992211', 'n02992529', 'n02999410', 'n03000134', 'n03000684', 'n03014705', 'n03016953', 'n03017168', 'n03018349', 'n03026506', 'n03028079', 'n03041632', 'n03045698', 'n03047690', 'n03062245', 'n03063599', 'n03063689', 'n03075370', 'n03095699', 'n03100240', 'n03109150', 'n03110669', 'n03124043', 'n03124170', 'n03125729', 'n03126707', 'n03127747', 'n03127925', 'n03131574', 'n03133878', 'n03134739', 'n03141823', 'n03146219', 'n03179701', 'n03187595', 'n03188531', 'n03196217', 'n03197337', 'n03201208', 'n03207743', 'n03207941', 'n03223299', 'n03240683', 'n03249569', 'n03250847', 'n03255030', 'n03259280', 'n03271574', 'n03272010', 'n03272562', 'n03291819', 'n03297495', 'n03314780', 'n03325584', 'n03337140', 'n03344393', 'n03345487', 'n03347037', 'n03355925', 'n03372029', 'n03376595', 'n03379051', 'n03384352', 'n03388183', 'n03388549', 'n03393912', 'n03394916', 'n03400231', 'n03404251', 'n03417042', 'n03424325', 'n03425413', 'n03443371', 'n03444034', 'n03445777', 'n03445924', 'n03447447', 'n03447721', 'n03450230', 'n03452741', 'n03467068', 'n03476684', 'n03476991', 'n03478589', 'n03481172', 'n03482405', 'n03483316', 'n03485407', 'n03485794', 'n03492542', 'n03494278', 'n03495258', 'n03496892', 'n03498962', 'n03527444', 'n03530642', 'n03532672', 'n03534580', 'n03535780', 'n03538406', 'n03544143', 'n03584254', 'n03584829', 'n03590841', 'n03594734', 'n03594945', 'n03595614', 'n03598930', 'n03599486', 'n03602883', 'n03617480', 'n03623198', 'n03627232', 'n03630383', 'n03633091', 'n03649909', 'n03657121', 'n03658185', 'n03662601', 'n03666591', 'n03670208', 'n03673027', 'n03676483', 'n03680355', 'n03690938', 'n03691459', 'n03692522', 'n03706229', 'n03709823', 'n03710193', 'n03717622', 'n03720891', 'n03721384', 'n03724870', 'n03729826', 'n03733131', 'n03733805', 'n03742115', 'n03743016', 'n03759954', 'n03761084', 'n03763968', 'n03764736', 'n03769881', 'n03770439', 'n03770679', 'n03773504', 'n03775071', 'n03775546', 'n03776460', 'n03777568', 'n03777754', 'n03781244', 'n03782006', 'n03785016', 'n03786901', 'n03787032', 'n03788195', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03793489', 'n03794056', 'n03796401', 'n03803284', 'n03804744', 'n03814639', 'n03814906', 'n03825788', 'n03832673', 'n03837869', 'n03838899', 'n03840681', 'n03841143', 'n03843555', 'n03854065', 'n03857828', 'n03866082', 'n03868242', 'n03868863', 'n03873416', 'n03874293', 'n03874599', 'n03876231', 'n03877472', 'n03877845', 'n03884397', 'n03888257', 'n03888605', 'n03891251', 'n03891332', 'n03895866', 'n03902125', 'n03903868', 'n03908618', 'n03908714', 'n03916031', 'n03920288', 'n03924679', 'n03929660', 'n03929855', 'n03930313', 'n03930630', 'n03933933', 'n03935335', 'n03937543', 'n03938244', 'n03942813', 'n03944341', 'n03947888', 'n03950228', 'n03954731', 'n03956157', 'n03958227', 'n03961711', 'n03970156', 'n03976467', 'n03976657', 'n03977966', 'n03980874', 'n03982430', 'n03983396', 'n03991062', 'n03992509', 'n03995372', 'n03998194', 'n04004767', 'n04005630', 'n04009552', 'n04019541', 'n04023962', 'n04026417', 'n04033901', 'n04033995', 'n04037443', 'n04039381', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04065272', 'n04067472', 'n04069434', 'n04070727', 'n04074963', 'n04086273', 'n04090263', 'n04099969', 'n04116512', 'n04118538', 'n04118776', 'n04120489', 'n04125021', 'n04127249', 'n04131690', 'n04133789', 'n04136333', 'n04141076', 'n04141327', 'n04141975', 'n04146614', 'n04147183', 'n04149813', 'n04153751', 'n04154565', 'n04162706', 'n04179913', 'n04192698', 'n04201297', 'n04204238', 'n04204347', 'n04208210', 'n04209133', 'n04209239', 'n04228054', 'n04229816', 'n04235860', 'n04238763', 'n04243546', 'n04252077', 'n04252225', 'n04254120', 'n04254680', 'n04254777', 'n04258138', 'n04259630', 'n04263257', 'n04265275', 'n04266014', 'n04270147', 'n04273569', 'n04275548', 'n04277352', 'n04285008', 'n04286575', 'n04310018', 'n04311004', 'n04311174', 'n04317175', 'n04325704', 'n04326547', 'n04328186', 'n04330267', 'n04332243', 'n04335435', 'n04336792', 'n04344873', 'n04346328', 'n04347754', 'n04350905', 'n04355338', 'n04355933', 'n04366367', 'n04367480', 'n04370456', 'n04371430', 'n04371774', 'n04376876', 'n04380533', 'n04389033', 'n04398044', 'n04399382', 'n04404412', 'n04409515', 'n04417672', 'n04418357', 'n04423845', 'n04428191', 'n04429376', 'n04435653', 'n04442312', 'n04447861', 'n04456115', 'n04458633', 'n04461696', 'n04465501', 'n04467665', 'n04476259', 'n04479046', 'n04482393', 'n04483307', 'n04485082', 'n04486054', 'n04487081', 'n04487394', 'n04505470', 'n04507155', 'n04509417', 'n04515003', 'n04517823', 'n04522168', 'n04525038', 'n04525305', 'n04532106', 'n04532670', 'n04536866', 'n04540053', 'n04542943', 'n04548280', 'n04548362', 'n04550184', 'n04552348', 'n04553703', 'n04554684', 'n04557648', 'n04560804', 'n04562935', 'n04579145', 'n04579432', 'n04584207', 'n04589890', 'n04590129', 'n04591157', 'n04591713', 'n04596742', 'n04597913', 'n04604644', 'n04606251', 'n04612504', 'n04613696', 'n06596364', 'n06794110', 'n06874185', 'n07565083', 'n07579787', 'n07583066', 'n07584110', 'n07613480', 'n07614500', 'n07615774', 'n07684084', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07711569', 'n07714571', 'n07714990', 'n07715103', 'n07716358', 'n07716906', 'n07717410', 'n07717556', 'n07718472', 'n07718747', 'n07720875', 'n07730033', 'n07734744', 'n07742313', 'n07745940', 'n07747607', 'n07749582', 'n07753113', 'n07753275', 'n07753592', 'n07754684', 'n07760859', 'n07768694', 'n07802026', 'n07831146', 'n07860988', 'n07871810', 'n07873807', 'n07875152', 'n07880968', 'n07930864', 'n07932039', 'n09229709', 'n09246464', 'n09256479', 'n09835506', 'n10565667', 'n11879895', 'n11939491', 'n12057211', 'n12144580', 'n12267677', 'n12620546', 'n12768682', 'n12985857', 'n12998815', 'n13037406', 'n13040303', 'n13044778', 'n13052670', 'n13054560', 'n13133613', 'n15075141']

IMAGENETS_FOLDER_NAMES = IMAGENETS_919_FOLDER_NAMES

IMAGENETS_300_FOLDER_NAMES = ['n01440764', 'n01443537', 'n01491361', 'n01494475', 'n01496331', 'n01518878', 'n01531178', 'n01532829', 'n01537544', 'n01608432', 'n01630670', 'n01632777', 'n01644373', 'n01644900', 'n01667114', 'n01675722', 'n01682714', 'n01685808', 'n01694178', 'n01695060', 'n01698640', 'n01704323', 'n01728572', 'n01728920', 'n01734418', 'n01744401', 'n01753488', 'n01770081', 'n01770393', 'n01773797', 'n01776313', 'n01817953', 'n01820546', 'n01855032', 'n01877812', 'n01882714', 'n01910747', 'n01914609', 'n01943899', 'n01980166', 'n01983481', 'n01984695', 'n01990800', 'n02011460', 'n02012849', 'n02013706', 'n02018795', 'n02058221', 'n02087046', 'n02088094', 'n02088632', 'n02090622', 'n02090721', 'n02091134', 'n02091244', 'n02093256', 'n02093754', 'n02094433', 'n02095570', 'n02097130', 'n02097209', 'n02097298', 'n02098413', 'n02100735', 'n02101556', 'n02102040', 'n02102177', 'n02104029', 'n02105412', 'n02107142', 'n02107312', 'n02110063', 'n02110958', 'n02111129', 'n02111500', 'n02111889', 'n02112706', 'n02113186', 'n02114855', 'n02119022', 'n02119789', 'n02120505', 'n02123394', 'n02123597', 'n02125311', 'n02127052', 'n02129604', 'n02133161', 'n02134418', 'n02165456', 'n02169497', 'n02177972', 'n02206856', 'n02256656', 'n02259212', 'n02268853', 'n02277742', 'n02280649', 'n02281406', 'n02321529', 'n02325366', 'n02326432', 'n02342885', 'n02396427', 'n02398521', 'n02415577', 'n02417914', 'n02447366', 'n02457408', 'n02480495', 'n02483362', 'n02488702', 'n02493793', 'n02494079', 'n02497673', 'n02504013', 'n02504458', 'n02510455', 'n02514041', 'n02526121', 'n02536864', 'n02669723', 'n02672831', 'n02690373', 'n02701002', 'n02708093', 'n02747177', 'n02769748', 'n02782093', 'n02783161', 'n02790996', 'n02793495', 'n02804610', 'n02808304', 'n02814533', 'n02835271', 'n02841315', 'n02859443', 'n02869837', 'n02870880', 'n02879718', 'n02892201', 'n02895154', 'n02917067', 'n02950826', 'n02951585', 'n02966687', 'n02978881', 'n02992529', 'n03000684', 'n03014705', 'n03018349', 'n03047690', 'n03075370', 'n03095699', 'n03109150', 'n03127925', 'n03133878', 'n03197337', 'n03201208', 'n03207941', 'n03223299', 'n03259280', 'n03271574', 'n03272562', 'n03291819', 'n03337140', 'n03376595', 'n03379051', 'n03393912', 'n03394916', 'n03404251', 'n03417042', 'n03443371', 'n03445777', 'n03452741', 'n03478589', 'n03482405', 'n03492542', 'n03494278', 'n03496892', 'n03532672', 'n03535780', 'n03538406', 'n03584829', 'n03590841', 'n03630383', 'n03633091', 'n03658185', 'n03673027', 'n03710193', 'n03743016', 'n03763968', 'n03764736', 'n03775546', 'n03781244', 'n03786901', 'n03788365', 'n03791053', 'n03792782', 'n03792972', 'n03794056', 'n03814906', 'n03825788', 'n03840681', 'n03874599', 'n03877845', 'n03888605', 'n03891251', 'n03903868', 'n03908714', 'n03929855', 'n03938244', 'n03956157', 'n03958227', 'n03976467', 'n03976657', 'n03991062', 'n04026417', 'n04033995', 'n04040759', 'n04041544', 'n04044716', 'n04049303', 'n04069434', 'n04070727', 'n04090263', 'n04099969', 'n04116512', 'n04118776', 'n04120489', 'n04179913', 'n04192698', 'n04201297', 'n04228054', 'n04229816', 'n04243546', 'n04254120', 'n04254680', 'n04254777', 'n04263257', 'n04265275', 'n04275548', 'n04277352', 'n04285008', 'n04311004', 'n04317175', 'n04335435', 'n04347754', 'n04371430', 'n04376876', 'n04380533', 'n04389033', 'n04399382', 'n04404412', 'n04429376', 'n04435653', 'n04447861', 'n04479046', 'n04483307', 'n04505470', 'n04507155', 'n04522168', 'n04540053', 'n04550184', 'n04552348', 'n04554684', 'n04557648', 'n04562935', 'n04579145', 'n04584207', 'n04591713', 'n04596742', 'n04606251', 'n04612504', 'n04613696', 'n06794110', 'n06874185', 'n07584110', 'n07614500', 'n07693725', 'n07697313', 'n07697537', 'n07711569', 'n07716906', 'n07720875', 'n07730033', 'n07742313', 'n07745940', 'n07749582', 'n07831146', 'n07880968', 'n07930864', 'n09256479', 'n12057211', 'n12768682', 'n12998815', 'n13037406', 'n13044778', 'n13054560']

IMAGENETS_50_FOLDER_NAMES = ['n01443537', 'n01491361', 'n01531178', 'n01644373', 'n02104029', 'n02119022', 'n02123597', 'n02133161', 'n02165456', 'n02281406', 'n02325366', 'n02342885', 'n02396427', 'n02483362', 'n02504458', 'n02510455', 'n02690373', 'n02747177', 'n02783161', 'n02814533', 'n02859443', 'n02917067', 'n02992529', 'n03014705', 'n03047690', 'n03095699', 'n03197337', 'n03201208', 'n03445777', 'n03452741', 'n03584829', 'n03630383', 'n03775546', 'n03791053', 'n03874599', 'n03891251', 'n04026417', 'n04335435', 'n04380533', 'n04404412', 'n04447861', 'n04507155', 'n04522168', 'n04557648', 'n04562935', 'n04612504', 'n06794110', 'n07749582', 'n07831146', 'n12998815']


IMAGENET_DEFAULT_TEMPLATES = [
    '{}.',
    'a bad photo of a {}.',
    'a photo of many {}.',
    'a sculpture of a {}.',
    'a photo of the hard to see {}.',
    'a low resolution photo of the {}.',
    'a rendering of a {}.',
    'graffiti of a {}.',
    'a bad photo of the {}.',
    'a cropped photo of the {}.',
    'a tattoo of a {}.',
    'the embroidered {}.',
    'a photo of a hard to see {}.',
    'a bright photo of a {}.',
    'a photo of a clean {}.',
    'a photo of a dirty {}.',
    'a dark photo of the {}.',
    'a drawing of a {}.',
    'a photo of my {}.',
    'the plastic {}.',
    'a photo of the cool {}.',
    'a close-up photo of a {}.',
    'a black and white photo of the {}.',
    'a painting of the {}.',
    'a painting of a {}.',
    'a pixelated photo of the {}.',
    'a sculpture of the {}.',
    'a bright photo of the {}.',
    'a cropped photo of a {}.',
    'a plastic {}.',
    'a photo of the dirty {}.',
    'a jpeg corrupted photo of a {}.',
    'a blurry photo of the {}.',
    'a photo of the {}.',
    'a good photo of the {}.',
    'a rendering of the {}.',
    'a {} in a video game.',
    'a photo of one {}.',
    'a doodle of a {}.',
    'a close-up photo of the {}.',
    'a photo of a {}.',
    'the origami {}.',
    'the {} in a video game.',
    'a sketch of a {}.',
    'a doodle of the {}.',
    'a origami {}.',
    'a low resolution photo of a {}.',
    'the toy {}.',
    'a rendition of the {}.',
    'a photo of the clean {}.',
    'a photo of a large {}.',
    'a rendition of a {}.',
    'a photo of a nice {}.',
    'a photo of a weird {}.',
    'a blurry photo of a {}.',
    'a cartoon {}.',
    'art of a {}.',
    'a sketch of the {}.',
    'a embroidered {}.',
    'a pixelated photo of a {}.',
    'itap of the {}.',
    'a jpeg corrupted photo of the {}.',
    'a good photo of a {}.',
    'a plushie {}.',
    'a photo of the nice {}.',
    'a photo of the small {}.',
    'a photo of the weird {}.',
    'the cartoon {}.',
    'art of the {}.',
    'a drawing of the {}.',
    'a photo of the large {}.',
    'a black and white photo of a {}.',
    'the plushie {}.',
    'a dark photo of a {}.',
    'itap of a {}.',
    'graffiti of the {}.',
    'a toy {}.',
    'itap of my {}.',
    'a photo of a cool {}.',
    'a photo of a small {}.',
    'a tattoo of the {}.',
]

IMAGENET_SIMPLE_TEMPLATES = [
    'a photo of {}.',
]
COCO_INSTANCE_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
COCO_PANOPTIC_CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush', 'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house', 'light', 'mirror-stuff', 'net', 'pillow', 'platform', 'playingfield', 'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow', 'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged', 'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged', 'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged', 'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']

COCO_IMAGENET_INDEX_PAIR = [[0, 591], [1, 444], [2, 479], [3, 671], [4, 405], [5, 779], [6, 466], [7, 717], [8, 472], [9, 920], [10, 686], [11, 920], [12, 704], [13, 708], [14, 134], [15, 282], [16, 215], [17, 291], [18, 349], [19, 341], [20, 366], [21, 294], [22, 340], [23, 286], [24, 414], [25, 879], [26, 636], [27, 906], [28, 519], [29, 162], [30, 795], [31, 671], [32, 852], [33, 405], [34, 805], [35, 560], [36, 671], [37, 671], [38, 752], [39, 907], [40, 907], [41, 787], [42, 792], [43, 792], [44, 792], [45, 538], [46, 954], [47, 948], [48, 697], [49, 950], [50, 937], [51, 936], [52, 934], [53, 963], [54, 931], [55, 415], [56, 423], [57, 831], [58, 738], [59, 520], [60, 532], [61, 896], [62, 851], [63, 620], [64, 674], [65, 761], [66, 508], [67, 487], [68, 447], [69, 827], [70, 859], [71, 896], [72, 760], [73, 454], [74, 892], [75, 883], [76, 591], [77, 850], [78, 584], [79, 696], [80, 879], [81, 672], [82, 839], [83, 519], [84, 896], [85, 854], [86, 799], [87, 858], [88, 947], [89, 953], [90, 792], [91, 425], [92, 862], [93, 475], [94, 916], [95, 721], [96, 538], [97, 703], [98, 705], [99, 979], [100, 888], [101, 538], [102, 774], [103, 390], [104, 519], [105, 803], [106, 708], [107, 672], [108, 879], [109, 825], [110, 825], [111, 858], [112, 825], [113, 898], [114, 904], [115, 904], [116, 947], [117, 862], [118, 858], [119, 862], [120, 648], [121, 736], [122, 904], [123, 708], [124, 970], [125, 936], [126, 792], [127, 549], [128, 712], [129, 647], [130, 972], [131, 904], [132, 824]]

PASCAL_CLASSES = [
    "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
    "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person",
    "pottedplant", "sheep", "sofa", "train", "tvmonitor"
]

# PASCAL_CLASSES = [
#     "airplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat",
#     "chair", "cow", "dining table", "dog", "horse", "motorcycle", "person",
#     "potted plant", "sheep", "couch", "train", "tv"
# ]

PASCAL_LABELS = [
                [0, 0, 0],
                [128, 0, 0],
                [0, 128, 0],
                [128, 128, 0],
                [0, 0, 128],
                [128, 0, 128],
                [0, 128, 128],
                [128, 128, 128],
                [64, 0, 0],
                [192, 0, 0],
                [64, 128, 0],
                [192, 128, 0],
                [64, 0, 128],
                [192, 0, 128],
                [64, 128, 128],
                [192, 128, 128],
                [0, 64, 0],
                [128, 64, 0],
                [0, 192, 0],
                [128, 192, 0],
                [0, 64, 128],
            ]

# ADE_PANOPTIC_CLASSES = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window ', 'grass', 'cabinet', 'sidewalk', 'person', 'ground', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'picture', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'closet', 'light', 'tub', 'rail', 'cushion', 'pedestal', 'box', 'column', 'signboard', 'dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm tree', 'kitchen island', 'computer', 'swivel chair', 'boat', 'pub', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'ceiling light', 'awning', 'street light', 'booth', 'tv', 'airplane', 'dirt road', 'clothes', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'transporter', 'canopy', 'washer', 'plaything', 'pool', 'stool', 'cylinder', 'basket', 'falls', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'stair', 'storage tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'exhaust hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'tvmonitor', 'bulletin board', 'shower', 'heater', 'drinking glass', 'clock', 'flag']

# ADE_PANOPTIC_CLASSES = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window ', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'closet', 'lamp', 'tub', 'rail', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm tree', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'street lamp', 'booth', 'tv', 'airplane', 'dirt road', 'clothes', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'pool', 'stool', 'barrel', 'basket', 'falls', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']


ADE_PANOPTIC_CLASSES = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'tub', 'rail', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'street lamp', 'booth', 'tv', 'plane', 'dirt track', 'clothes', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'pool', 'stool', 'barrel', 'basket', 'falls', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag']


COCO_INTER_ADE = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed', 'window', 'grass', 'cabinet', 'pavement', 'person', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'shelf', 'house', 'sea', 'mirror', 'rug', 'fence', 'rock', 'stone', 'sign', 'counter', 'sand', 'sink', 'refrigerator', 'stairs', 'table', 'pillow', 'door', 'river', 'bridge', 'blind', 'table', 'toilet', 'flower', 'book', 'bench', 'tree', 'chair', 'boat', 'bus', 'towel', 'light', 'truck', 'tv', 'dirt', 'bottle', 'counter', 'tent', 'oven', 'ball', 'food', 'microwave', 'bicycle', 'blanket', 'vase', 'traffic', 'light', 'glass', 'clock']

PASCAL_CONTEXT_459 = ['accordion', 'aeroplane', 'air conditioner', 'antenna', 'artillery', 'ashtray', 'atrium', 'baby carriage', 'bag', 'ball', 'balloon', 'bamboo weaving', 'barrel', 'baseball bat', 'basket', 'basketball backboard', 'bathtub', 'bed', 'bedclothes', 'beer', 'bell', 'bench', 'bicycle', 'binoculars', 'bird', 'bird cage', 'bird feeder', 'bird nest', 'blackboard', 'board', 'boat', 'bone', 'book', 'bottle', 'bottle opener', 'bowl', 'box', 'bracelet', 'brick', 'bridge', 'broom', 'brush', 'bucket', 'building', 'bus', 'cabinet', 'cabinet door', 'cage', 'cake', 'calculator', 'calendar', 'camel', 'camera', 'camera lens', 'can', 'candle', 'candle holder', 'cap', 'car', 'card', 'cart', 'case', 'casette recorder', 'cash register', 'cat', 'cd', 'cd player', 'ceiling', 'cell phone', 'cello', 'chain', 'chair', 'chessboard', 'chicken', 'chopstick', 'clip', 'clippers', 'clock', 'closet', 'cloth', 'clothes tree', 'coffee', 'coffee machine', 'comb', 'computer', 'concrete', 'cone', 'container', 'control booth', 'controller', 'cooker', 'copying machine', 'coral', 'cork', 'corkscrew', 'counter', 'court', 'cow', 'crabstick', 'crane', 'crate', 'cross', 'crutch', 'cup', 'curtain', 'cushion', 'cutting board', 'dais', 'disc', 'disc case', 'dishwasher', 'dock', 'dog', 'dolphin', 'door', 'drainer', 'dray', 'drink dispenser', 'drinking machine', 'drop', 'drug', 'drum', 'drum kit', 'duck', 'dumbbell', 'earphone', 'earrings', 'egg', 'electric fan', 'electric iron', 'electric pot', 'electric saw', 'electronic keyboard', 'engine', 'envelope', 'equipment', 'escalator', 'exhibition booth', 'extinguisher', 'eyeglass', 'fan', 'faucet', 'fax machine', 'fence', 'ferris wheel', 'fire extinguisher', 'fire hydrant', 'fire place', 'fish', 'fish tank', 'fishbowl', 'fishing net', 'fishing pole', 'flag', 'flagstaff', 'flame', 'flashlight', 'floor', 'flower', 'fly', 'foam', 'food', 'footbridge', 'forceps', 'fork', 'forklift', 'fountain', 'fox', 'frame', 'fridge', 'frog', 'fruit', 'funnel', 'furnace', 'game controller', 'game machine', 'gas cylinder', 'gas hood', 'gas stove', 'gift box', 'glass', 'glass marble', 'globe', 'glove', 'goal', 'grandstand', 'grass', 'gravestone', 'ground', 'guardrail', 'guitar', 'gun', 'hammer', 'hand cart', 'handle', 'handrail', 'hanger', 'hard disk drive', 'hat', 'hay', 'headphone', 'heater', 'helicopter', 'helmet', 'holder', 'hook', 'horse', 'horse-drawn carriage', 'hot-air balloon', 'hydrovalve', 'ice', 'inflator pump', 'ipod', 'iron', 'ironing board', 'jar', 'kart', 'kettle', 'key', 'keyboard', 'kitchen range', 'kite', 'knife', 'knife block', 'ladder', 'ladder truck', 'ladle', 'laptop', 'leaves', 'lid', 'life buoy', 'light', 'light bulb', 'lighter', 'line', 'lion', 'lobster', 'lock', 'machine', 'mailbox', 'mannequin', 'map', 'mask', 'mat', 'match book', 'mattress', 'menu', 'metal', 'meter box', 'microphone', 'microwave', 'mirror', 'missile', 'model', 'money', 'monkey', 'mop', 'motorbike', 'mountain', 'mouse', 'mouse pad', 'musical instrument', 'napkin', 'net', 'newspaper', 'oar', 'ornament', 'outlet', 'oven', 'oxygen bottle', 'pack', 'pan', 'paper', 'paper box', 'paper cutter', 'parachute', 'parasol', 'parterre', 'patio', 'pelage', 'pen', 'pen container', 'pencil', 'person', 'photo', 'piano', 'picture', 'pig', 'pillar', 'pillow', 'pipe', 'pitcher', 'plant', 'plastic', 'plate', 'platform', 'player', 'playground', 'pliers', 'plume', 'poker', 'poker chip', 'pole', 'pool table', 'postcard', 'poster', 'pot', 'pottedplant', 'printer', 'projector', 'pumpkin', 'rabbit', 'racket', 'radiator', 'radio', 'rail', 'rake', 'ramp', 'range hood', 'receiver', 'recorder', 'recreational machines', 'remote control', 'road', 'robot', 'rock', 'rocket', 'rocking horse', 'rope', 'rug', 'ruler', 'runway', 'saddle', 'sand', 'saw', 'scale', 'scanner', 'scissors', 'scoop', 'screen', 'screwdriver', 'sculpture', 'scythe', 'sewer', 'sewing machine', 'shed', 'sheep', 'shell', 'shelves', 'shoe', 'shopping cart', 'shovel', 'sidecar', 'sidewalk', 'sign', 'signal light', 'sink', 'skateboard', 'ski', 'sky', 'sled', 'slippers', 'smoke', 'snail', 'snake', 'snow', 'snowmobiles', 'sofa', 'spanner', 'spatula', 'speaker', 'speed bump', 'spice container', 'spoon', 'sprayer', 'squirrel', 'stage', 'stair', 'stapler', 'stick', 'sticky note', 'stone', 'stool', 'stove', 'straw', 'stretcher', 'sun', 'sunglass', 'sunshade', 'surveillance camera', 'swan', 'sweeper', 'swim ring', 'swimming pool', 'swing', 'switch', 'table', 'tableware', 'tank', 'tap', 'tape', 'tarp', 'telephone', 'telephone booth', 'tent', 'tire', 'toaster', 'toilet', 'tong', 'tool', 'toothbrush', 'towel', 'toy', 'toy car', 'track', 'train', 'trampoline', 'trash bin', 'tray', 'tree', 'tricycle', 'tripod', 'trophy', 'truck', 'tube', 'turtle', 'tvmonitor', 'tweezers', 'typewriter', 'umbrella', 'unknown', 'vacuum cleaner', 'vending machine', 'video camera', 'video game console', 'video player', 'video tape', 'violin', 'wakeboard', 'wall', 'wallet', 'wardrobe', 'washing machine', 'watch', 'water', 'water dispenser', 'water pipe', 'water skate board', 'watermelon', 'whale', 'wharf', 'wheel', 'wheelchair', 'window', 'window blinds', 'wineglass', 'wire', 'wood', 'wool']

PASCAL_CONTEXT_33 = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor', 'sky', 'grass', 'ground', 'road', 'building', 'tree', 'water', 'mountain', 'wall', 'floor', 'track', 'keyboard', 'ceiling']

PASCAL_CONTEXT_59 = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'dining table', 'dog', 'horse', 'motorbike', 'person', 'potted plant', 'sheep', 'sofa', 'train', 'tv monitor', 'bag', 'bed', 'bench', 'book', 'building', 'cabinet', 'ceiling', 'cloth', 'computer', 'cup', 'door', 'fence', 'floor', 'flower', 'food', 'grass', 'ground', 'keyboard', 'light', 'mountain', 'mouse', 'curtain', 'platform', 'sign', 'plate', 'road', 'rock', 'shelves', 'side walk', 'sky', 'snow', 'bed clothes', 'track', 'tree', 'truck', 'wall', 'water', 'window', 'wood']

SUN_RGBD_37 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

SCAN_37 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag']

SCAN_40 = ['wall', 'floor', 'cabinet', 'bed', 'chair', 'sofa', 'table', 'door', 'window', 'bookshelf', 'picture', 'counter', 'blinds', 'desk', 'shelves', 'curtain', 'dresser', 'pillow', 'mirror', 'floor mat', 'clothes', 'ceiling', 'books', 'refridgerator', 'television', 'paper', 'towel', 'shower curtain', 'box', 'whiteboard', 'person', 'night stand', 'toilet', 'sink', 'lamp', 'bathtub', 'bag', 'otherstructure', 'otherfurniture', 'otherprop']

SCAN_20 = ["wall", "floor", "cabinet", "bed", "chair", "sofa", "table", "door", "window", "bookshelf", "picture", "counter", "desk", "curtain", "refrigerator", "shower curtain", "toilet", "sink", "bathtub", "otherfurniture"]

CITYSCAPES = ['road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle']

CITYSCAPES_THING = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

BDD_SEM = ["road", "sidewalk", "building", "wall", "fence", "pole", "traffic light", "traffic sign", "vegetation", "terrain", "sky", "person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

BDD_PANO = ['dynamic', 'ego vehicle', 'ground', 'static', 'parking', 'rail track', 'road', 'sidewalk', 'bridge', 'building', 'fence', 'garage', 'guard rail', 'tunnel', 'wall', 'banner', 'billboard', 'lane divider', 'parking sign', 'pole', 'polegroup', 'street light', 'traffic cone', 'traffic device', 'traffic light', 'traffic sign', 'traffic sign frame', 'terrain', 'vegetation', 'sky', 'person', 'rider', 'bicycle', 'bus', 'car', 'caravan', 'motorcycle', 'trailer', 'train', 'truck']

OBJECT365 = ['Person', 'Sneakers', 'Chair', 'Other Shoes', 'Hat', 'Car', 'Lamp', 'Glasses', 'Bottle', 'Desk', 'Cup', 'Street Lights', 'Cabinet/shelf', 'Handbag/Satchel', 'Bracelet', 'Plate', 'Picture/Frame', 'Helmet', 'Book', 'Gloves', 'Storage box', 'Boat', 'Leather Shoes', 'Flower', 'Bench', 'Potted Plant', 'Bowl/Basin', 'Flag', 'Pillow', 'Boots', 'Vase', 'Microphone', 'Necklace', 'Ring', 'SUV', 'Wine Glass', 'Belt', 'Moniter/TV', 'Backpack', 'Umbrella', 'Traffic Light', 'Speaker', 'Watch', 'Tie', 'Trash bin Can', 'Slippers', 'Bicycle', 'Stool', 'Barrel/bucket', 'Van', 'Couch', 'Sandals', 'Bakset', 'Drum', 'Pen/Pencil', 'Bus', 'Wild Bird', 'High Heels', 'Motorcycle', 'Guitar', 'Carpet', 'Cell Phone', 'Bread', 'Camera', 'Canned', 'Truck', 'Traffic cone', 'Cymbal', 'Lifesaver', 'Towel', 'Stuffed Toy', 'Candle', 'Sailboat', 'Laptop', 'Awning', 'Bed', 'Faucet', 'Tent', 'Horse', 'Mirror', 'Power outlet', 'Sink', 'Apple', 'Air Conditioner', 'Knife', 'Hockey Stick', 'Paddle', 'Pickup Truck', 'Fork', 'Traffic Sign', 'Ballon', 'Tripod', 'Dog', 'Spoon', 'Clock', 'Pot', 'Cow', 'Cake', 'Dinning Table', 'Sheep', 'Hanger', 'Blackboard/Whiteboard', 'Napkin', 'Other Fish', 'Orange/Tangerine', 'Toiletry', 'Keyboard', 'Tomato', 'Lantern', 'Machinery Vehicle', 'Fan', 'Green Vegetables', 'Banana', 'Baseball Glove', 'Airplane', 'Mouse', 'Train', 'Pumpkin', 'Soccer', 'Skiboard', 'Luggage', 'Nightstand', 'Tea pot', 'Telephone', 'Trolley', 'Head Phone', 'Sports Car', 'Stop Sign', 'Dessert', 'Scooter', 'Stroller', 'Crane', 'Remote', 'Refrigerator', 'Oven', 'Lemon', 'Duck', 'Baseball Bat', 'Surveillance Camera', 'Cat', 'Jug', 'Broccoli', 'Piano', 'Pizza', 'Elephant', 'Skateboard', 'Surfboard', 'Gun', 'Skating and Skiing shoes', 'Gas stove', 'Donut', 'Bow Tie', 'Carrot', 'Toilet', 'Kite', 'Strawberry', 'Other Balls', 'Shovel', 'Pepper', 'Computer Box', 'Toilet Paper', 'Cleaning Products', 'Chopsticks', 'Microwave', 'Pigeon', 'Baseball', 'Cutting/chopping Board', 'Coffee Table', 'Side Table', 'Scissors', 'Marker', 'Pie', 'Ladder', 'Snowboard', 'Cookies', 'Radiator', 'Fire Hydrant', 'Basketball', 'Zebra', 'Grape', 'Giraffe', 'Potato', 'Sausage', 'Tricycle', 'Violin', 'Egg', 'Fire Extinguisher', 'Candy', 'Fire Truck', 'Billards', 'Converter', 'Bathtub', 'Wheelchair', 'Golf Club', 'Briefcase', 'Cucumber', 'Cigar/Cigarette ', 'Paint Brush', 'Pear', 'Heavy Truck', 'Hamburger', 'Extractor', 'Extention Cord', 'Tong', 'Tennis Racket', 'Folder', 'American Football', 'earphone', 'Mask', 'Kettle', 'Tennis', 'Ship', 'Swing', 'Coffee Machine', 'Slide', 'Carriage', 'Onion', 'Green beans', 'Projector', 'Frisbee', 'Washing Machine/Drying Machine', 'Chicken', 'Printer', 'Watermelon', 'Saxophone', 'Tissue', 'Toothbrush', 'Ice cream', 'Hotair ballon', 'Cello', 'French Fries', 'Scale', 'Trophy', 'Cabbage', 'Hot dog', 'Blender', 'Peach', 'Rice', 'Wallet/Purse', 'Volleyball', 'Deer', 'Goose', 'Tape', 'Tablet', 'Cosmetics', 'Trumpet', 'Pineapple', 'Golf Ball', 'Ambulance', 'Parking meter', 'Mango', 'Key', 'Hurdle', 'Fishing Rod', 'Medal', 'Flute', 'Brush', 'Penguin', 'Megaphone', 'Corn', 'Lettuce', 'Garlic', 'Swan', 'Helicopter', 'Green Onion', 'Sandwich', 'Nuts', 'Speed Limit Sign', 'Induction Cooker', 'Broom', 'Trombone', 'Plum', 'Rickshaw', 'Goldfish', 'Kiwi fruit', 'Router/modem', 'Poker Card', 'Toaster', 'Shrimp', 'Sushi', 'Cheese', 'Notepaper', 'Cherry', 'Pliers', 'CD', 'Pasta', 'Hammer', 'Cue', 'Avocado', 'Hamimelon', 'Flask', 'Mushroon', 'Screwdriver', 'Soap', 'Recorder', 'Bear', 'Eggplant', 'Board Eraser', 'Coconut', 'Tape Measur/ Ruler', 'Pig', 'Showerhead', 'Globe', 'Chips', 'Steak', 'Crosswalk Sign', 'Stapler', 'Campel', 'Formula 1 ', 'Pomegranate', 'Dishwasher', 'Crab', 'Hoverboard', 'Meat ball', 'Rice Cooker', 'Tuba', 'Calculator', 'Papaya', 'Antelope', 'Parrot', 'Seal', 'Buttefly', 'Dumbbell', 'Donkey', 'Lion', 'Urinal', 'Dolphin', 'Electric Drill', 'Hair Dryer', 'Egg tart', 'Jellyfish', 'Treadmill', 'Lighter', 'Grapefruit', 'Game board', 'Mop', 'Radish', 'Baozi', 'Target', 'French', 'Spring Rolls', 'Monkey', 'Rabbit', 'Pencil Case', 'Yak', 'Red Cabbage', 'Binoculars', 'Asparagus', 'Barbell', 'Scallop', 'Noddles', 'Comb', 'Dumpling', 'Oyster', 'Table Teniis paddle', 'Cosmetics Brush/Eyeliner Pencil', 'Chainsaw', 'Eraser', 'Lobster', 'Durian', 'Okra', 'Lipstick', 'Cosmetics Mirror', 'Curling', 'Table Tennis ']


OPENIMAGE = ['Tortoise', 'Container', 'Magpie', 'Sea turtle', 'Football', 'Ambulance', 'Ladder', 'Toothbrush', 'Syringe', 'Sink', 'Toy', 'Organ (Musical Instrument)', 'Cassette deck', 'Apple', 'Human eye', 'Cosmetics', 'Paddle', 'Snowman', 'Beer', 'Chopsticks', 'Human beard', 'Bird', 'Parking meter', 'Traffic light', 'Croissant', 'Cucumber', 'Radish', 'Towel', 'Doll', 'Skull', 'Washing machine', 'Glove', 'Tick', 'Belt', 'Sunglasses', 'Banjo', 'Cart', 'Ball', 'Backpack', 'Bicycle', 'Home appliance', 'Centipede', 'Boat', 'Surfboard', 'Boot', 'Headphones', 'Hot dog', 'Shorts', 'Fast food', 'Bus', 'Boy', 'Screwdriver', 'Bicycle wheel', 'Barge', 'Laptop', 'Miniskirt', 'Drill (Tool)', 'Dress', 'Bear', 'Waffle', 'Pancake', 'Brown bear', 'Woodpecker', 'Blue jay', 'Pretzel', 'Bagel', 'Tower', 'Teapot', 'Person', 'Bow and arrow', 'Swimwear', 'Beehive', 'Brassiere', 'Bee', 'Bat (Animal)', 'Starfish', 'Popcorn', 'Burrito', 'Chainsaw', 'Balloon', 'Wrench', 'Tent', 'Vehicle registration plate', 'Lantern', 'Toaster', 'Flashlight', 'Billboard', 'Tiara', 'Limousine', 'Necklace', 'Carnivore', 'Scissors', 'Stairs', 'Computer keyboard', 'Printer', 'Traffic sign', 'Chair', 'Shirt', 'Poster', 'Cheese', 'Sock', 'Fire hydrant', 'Land vehicle', 'Earrings', 'Tie', 'Watercraft', 'Cabinetry', 'Suitcase', 'Muffin', 'Bidet', 'Snack', 'Snowmobile', 'Clock', 'Medical equipment', 'Cattle', 'Cello', 'Jet ski', 'Camel', 'Coat', 'Suit', 'Desk', 'Cat', 'Bronze sculpture', 'Juice', 'Gondola', 'Beetle', 'Cannon', 'Computer mouse', 'Cookie', 'Office building', 'Fountain', 'Coin', 'Calculator', 'Cocktail', 'Computer monitor', 'Box', 'Stapler', 'Christmas tree', 'Cowboy hat', 'Hiking equipment', 'Studio couch', 'Drum', 'Dessert', 'Wine rack', 'Drink', 'Zucchini', 'Ladle', 'Human mouth', 'Dairy Product', 'Dice', 'Oven', 'Dinosaur', 'Ratchet (Device)', 'Couch', 'Cricket ball', 'Winter melon', 'Spatula', 'Whiteboard', 'Pencil sharpener', 'Door', 'Hat', 'Shower', 'Eraser', 'Fedora', 'Guacamole', 'Dagger', 'Scarf', 'Dolphin', 'Sombrero', 'Tin can', 'Mug', 'Tap', 'Harbor seal', 'Stretcher', 'Can opener', 'Goggles', 'Human body', 'Roller skates', 'Coffee cup', 'Cutting board', 'Blender', 'Plumbing fixture', 'Stop sign', 'Office supplies', 'Volleyball (Ball)', 'Vase', 'Slow cooker', 'Wardrobe', 'Coffee', 'Whisk', 'Paper towel', 'Personal care', 'Food', 'Sun hat', 'Tree house', 'Flying disc', 'Skirt', 'Gas stove', 'Salt and pepper shakers', 'Mechanical fan', 'Face powder', 'Fax', 'Fruit', 'French fries', 'Nightstand', 'Barrel', 'Kite', 'Tart', 'Treadmill', 'Fox', 'Flag', 'French horn', 'Window blind', 'Human foot', 'Golf cart', 'Jacket', 'Egg (Food)', 'Street light', 'Guitar', 'Pillow', 'Human leg', 'Isopod', 'Grape', 'Human ear', 'Power plugs and sockets', 'Panda', 'Giraffe', 'Woman', 'Door handle', 'Rhinoceros', 'Bathtub', 'Goldfish', 'Houseplant', 'Goat', 'Baseball bat', 'Baseball glove', 'Mixing bowl', 'Marine invertebrates', 'Kitchen utensil', 'Light switch', 'House', 'Horse', 'Stationary bicycle', 'Hammer', 'Ceiling fan', 'Sofa bed', 'Adhesive tape', 'Harp', 'Sandal', 'Bicycle helmet', 'Saucer', 'Harpsichord', 'Human hair', 'Heater', 'Harmonica', 'Hamster', 'Curtain', 'Bed', 'Kettle', 'Fireplace', 'Scale', 'Drinking straw', 'Insect', 'Hair dryer', 'Kitchenware', 'Indoor rower', 'Invertebrate', 'Food processor', 'Bookcase', 'Refrigerator', 'Wood-burning stove', 'Punching bag', 'Common fig', 'Cocktail shaker', 'Jaguar (Animal)', 'Golf ball', 'Fashion accessory', 'Alarm clock', 'Filing cabinet', 'Artichoke', 'Table', 'Tableware', 'Kangaroo', 'Koala', 'Knife', 'Bottle', 'Bottle opener', 'Lynx', 'Lavender (Plant)', 'Lighthouse', 'Dumbbell', 'Human head', 'Bowl', 'Humidifier', 'Porch', 'Lizard', 'Billiard table', 'Mammal', 'Mouse', 'Motorcycle', 'Musical instrument', 'Swim cap', 'Frying pan', 'Snowplow', 'Bathroom cabinet', 'Missile', 'Bust', 'Man', 'Waffle iron', 'Milk', 'Ring binder', 'Plate', 'Mobile phone', 'Baked goods', 'Mushroom', 'Crutch', 'Pitcher (Container)', 'Mirror', 'Personal flotation device', 'Table tennis racket', 'Pencil case', 'Musical keyboard', 'Scoreboard', 'Briefcase', 'Kitchen knife', 'Nail (Construction)', 'Tennis ball', 'Plastic bag', 'Oboe', 'Chest of drawers', 'Ostrich', 'Piano', 'Girl', 'Plant', 'Potato', 'Hair spray', 'Sports equipment', 'Pasta', 'Penguin', 'Pumpkin', 'Pear', 'Infant bed', 'Polar bear', 'Mixer', 'Cupboard', 'Jacuzzi', 'Pizza', 'Digital clock', 'Pig', 'Reptile', 'Rifle', 'Lipstick', 'Skateboard', 'Raven', 'High heels', 'Red panda', 'Rose', 'Rabbit', 'Sculpture', 'Saxophone', 'Shotgun', 'Seafood', 'Submarine sandwich', 'Snowboard', 'Sword', 'Picture frame', 'Sushi', 'Loveseat', 'Ski', 'Squirrel', 'Tripod', 'Stethoscope', 'Submarine', 'Scorpion', 'Segway', 'Training bench', 'Snake', 'Coffee table', 'Skyscraper', 'Sheep', 'Television', 'Trombone', 'Tea', 'Tank', 'Taco', 'Telephone', 'Torch', 'Tiger', 'Strawberry', 'Trumpet', 'Tree', 'Tomato', 'Train', 'Tool', 'Picnic basket', 'Cooking spray', 'Trousers', 'Bowling equipment', 'Football helmet', 'Truck', 'Measuring cup', 'Coffeemaker', 'Violin', 'Vehicle', 'Handbag', 'Paper cutter', 'Wine', 'Weapon', 'Wheel', 'Worm', 'Wok', 'Whale', 'Zebra', 'Auto part', 'Jug', 'Pizza cutter', 'Cream', 'Monkey', 'Lion', 'Bread', 'Platter', 'Chicken', 'Eagle', 'Helicopter', 'Owl', 'Duck', 'Turtle', 'Hippopotamus', 'Crocodile', 'Toilet', 'Toilet paper', 'Squid', 'Clothing', 'Footwear', 'Lemon', 'Spider', 'Deer', 'Frog', 'Banana', 'Rocket', 'Wine glass', 'Countertop', 'Tablet computer', 'Waste container', 'Swimming pool', 'Dog', 'Book', 'Elephant', 'Shark', 'Candle', 'Leopard', 'Axe', 'Hand dryer', 'Soap dispenser', 'Porcupine', 'Flower', 'Canary', 'Cheetah', 'Palm tree', 'Hamburger', 'Maple', 'Building', 'Fish', 'Lobster', 'Garden Asparagus', 'Furniture', 'Hedgehog', 'Airplane', 'Spoon', 'Otter', 'Bull', 'Oyster', 'Horizontal bar', 'Convenience store', 'Bomb', 'Bench', 'Ice cream', 'Caterpillar', 'Butterfly', 'Parachute', 'Orange', 'Antelope', 'Beaker', 'Moths and butterflies', 'Window', 'Closet', 'Castle', 'Jellyfish', 'Goose', 'Mule', 'Swan', 'Peach', 'Coconut', 'Seat belt', 'Raccoon', 'Chisel', 'Fork', 'Lamp', 'Camera', 'Squash (Plant)', 'Racket', 'Human face', 'Human arm', 'Vegetable', 'Diaper', 'Unicycle', 'Falcon', 'Chime', 'Snail', 'Shellfish', 'Cabbage', 'Carrot', 'Mango', 'Jeans', 'Flowerpot', 'Pineapple', 'Drawer', 'Stool', 'Envelope', 'Cake', 'Dragonfly', 'Common sunflower', 'Microwave oven', 'Honeycomb', 'Marine mammal', 'Sea lion', 'Ladybug', 'Shelf', 'Watch', 'Candy', 'Salad', 'Parrot', 'Handgun', 'Sparrow', 'Van', 'Grinder', 'Spice rack', 'Light bulb', 'Corded phone', 'Sports uniform', 'Tennis racket', 'Wall clock', 'Serving tray', 'Kitchen & dining room table', 'Dog bed', 'Cake stand', 'Cat furniture', 'Bathroom accessory', 'Facial tissue holder', 'Pressure cooker', 'Kitchen appliance', 'Tire', 'Ruler', 'Luggage and bags', 'Microphone', 'Broccoli', 'Umbrella', 'Pastry', 'Grapefruit', 'Band-aid', 'Animal', 'Bell pepper', 'Turkey', 'Lily', 'Pomegranate', 'Doughnut', 'Glasses', 'Human nose', 'Pen', 'Ant', 'Car', 'Aircraft', 'Human hand', 'Skunk', 'Teddy bear', 'Watermelon', 'Cantaloupe', 'Dishwasher', 'Flute', 'Balance beam', 'Sandwich', 'Shrimp', 'Sewing machine', 'Binoculars', 'Rays and skates', 'Ipod', 'Accordion', 'Willow', 'Crab', 'Crown', 'Seahorse', 'Perfume', 'Alpaca', 'Taxi', 'Canoe', 'Remote control', 'Wheelchair', 'Rugby ball', 'Armadillo', 'Maracas', 'Helmet']

ADE20K_847 = ['wall', 'building', 'sky', 'tree', 'road', 'floor', 'ceiling', 'bed', 'sidewalk', 'earth', 'cabinet', 'person', 'grass', 'windowpane', 'car', 'mountain', 'plant', 'table', 'chair', 'curtain', 'door', 'sofa', 'sea', 'painting', 'water', 'mirror', 'house', 'rug', 'shelf', 'armchair', 'fence', 'field', 'lamp', 'rock', 'seat', 'river', 'desk', 'bathtub', 'railing', 'signboard', 'cushion', 'path', 'work surface', 'stairs', 'column', 'sink', 'wardrobe', 'snow', 'refrigerator', 'base', 'bridge', 'blind', 'runway', 'cliff', 'sand', 'fireplace', 'pillow', 'screen door', 'toilet', 'skyscraper', 'grandstand', 'box', 'pool table', 'palm', 'double door', 'coffee table', 'counter', 'countertop', 'chest of drawers', 'kitchen island', 'boat', 'waterfall', 'stove', 'flower', 'bookcase', 'controls', 'book', 'stairway', 'streetlight', 'computer', 'bus', 'swivel chair', 'light', 'bench', 'case', 'towel', 'fountain', 'embankment', 'television receiver', 'van', 'hill', 'awning', 'poster', 'truck', 'airplane', 'pole', 'tower', 'court', 'ball', 'aircraft carrier', 'buffet', 'hovel', 'apparel', 'minibike', 'animal', 'chandelier', 'step', 'booth', 'bicycle', 'doorframe', 'sconce', 'pond', 'trade name', 'bannister', 'bag', 'traffic light', 'gazebo', 'escalator', 'land', 'board', 'arcade machine', 'eiderdown', 'bar', 'stall', 'playground', 'ship', 'ottoman', 'ashcan', 'bottle', 'cradle', 'pot', 'conveyer belt', 'train', 'stool', 'lake', 'tank', 'ice', 'basket', 'manhole', 'tent', 'canopy', 'microwave', 'barrel', 'dirt track', 'beam', 'dishwasher', 'plate', 'screen', 'ruins', 'washer', 'blanket', 'plaything', 'food', 'screen', 'oven', 'stage', 'beacon', 'umbrella', 'sculpture', 'aqueduct', 'container', 'scaffolding', 'hood', 'curb', 'roller coaster', 'horse', 'catwalk', 'glass', 'vase', 'central reservation', 'carousel', 'radiator', 'closet', 'machine', 'pier', 'fan', 'inflatable bounce game', 'pitch', 'paper', 'arcade', 'hot tub', 'helicopter', 'tray', 'partition', 'vineyard', 'bowl', 'bullring', 'flag', 'pot', 'footbridge', 'shower', 'bag', 'bulletin board', 'confessional booth', 'trunk', 'forest', 'elevator door', 'laptop', 'instrument panel', 'bucket', 'tapestry', 'platform', 'jacket', 'gate', 'monitor', 'telephone booth', 'spotlight', 'ring', 'control panel', 'blackboard', 'air conditioner', 'chest', 'clock', 'sand dune', 'pipe', 'vault', 'table football', 'cannon', 'swimming pool', 'fluorescent', 'statue', 'loudspeaker', 'exhibitor', 'ladder', 'carport', 'dam', 'pulpit', 'skylight', 'water tower', 'grill', 'display board', 'pane', 'rubbish', 'ice rink', 'fruit', 'patio', 'vending machine', 'telephone', 'net', 'backpack', 'jar', 'track', 'magazine', 'shutter', 'roof', 'banner', 'landfill', 'post', 'altarpiece', 'hat', 'arch', 'table game', 'bag', 'document', 'dome', 'pier', 'shanties', 'forecourt', 'crane', 'dog', 'piano', 'drawing', 'cabin', 'ad', 'amphitheater', 'monument', 'henhouse', 'cockpit', 'heater', 'windmill', 'pool', 'elevator', 'decoration', 'labyrinth', 'text', 'printer', 'mezzanine', 'mattress', 'straw', 'stalls', 'patio', 'billboard', 'bus stop', 'trouser', 'console table', 'rack', 'notebook', 'shrine', 'pantry', 'cart', 'steam shovel', 'porch', 'postbox', 'figurine', 'recycling bin', 'folding screen', 'telescope', 'deck chair', 'kennel', 'coffee maker', 'altar', 'fish', 'easel', 'artificial golf green', 'iceberg', 'candlestick', 'shower stall', 'television stand', 'wall socket', 'skeleton', 'grand piano', 'candy', 'grille door', 'pedestal', 'jersey', 'shoe', 'gravestone', 'shanty', 'structure', 'rocking chair', 'bird', 'place mat', 'tomb', 'big top', 'gas pump', 'lockers', 'cage', 'finger', 'bleachers', 'ferris wheel', 'hairdresser chair', 'mat', 'stands', 'aquarium', 'streetcar', 'napkin', 'dummy', 'booklet', 'sand trap', 'shop', 'table cloth', 'service station', 'coffin', 'drawer', 'cages', 'slot machine', 'balcony', 'volleyball court', 'table tennis', 'control table', 'shirt', 'merchandise', 'railway', 'parterre', 'chimney', 'can', 'tanks', 'fabric', 'alga', 'system', 'map', 'greenhouse', 'mug', 'barbecue', 'trailer', 'toilet tissue', 'organ', 'dishrag', 'island', 'keyboard', 'trench', 'basket', 'steering wheel', 'pitcher', 'goal', 'bread', 'beds', 'wood', 'file cabinet', 'newspaper', 'motorboat', 'rope', 'guitar', 'rubble', 'scarf', 'barrels', 'cap', 'leaves', 'control tower', 'dashboard', 'bandstand', 'lectern', 'switch', 'baseboard', 'shower room', 'smoke', 'faucet', 'bulldozer', 'saucepan', 'shops', 'meter', 'crevasse', 'gear', 'candelabrum', 'sofa bed', 'tunnel', 'pallet', 'wire', 'kettle', 'bidet', 'baby buggy', 'music stand', 'pipe', 'cup', 'parking meter', 'ice hockey rink', 'shelter', 'weeds', 'temple', 'patty', 'ski slope', 'panel', 'wallet', 'wheel', 'towel rack', 'roundabout', 'canister', 'rod', 'soap dispenser', 'bell', 'canvas', 'box office', 'teacup', 'trellis', 'workbench', 'valley', 'toaster', 'knife', 'podium', 'ramp', 'tumble dryer', 'fireplug', 'gym shoe', 'lab bench', 'equipment', 'rocky formation', 'plastic', 'calendar', 'caravan', 'check-in-desk', 'ticket counter', 'brush', 'mill', 'covered bridge', 'bowling alley', 'hanger', 'excavator', 'trestle', 'revolving door', 'blast furnace', 'scale', 'projector', 'soap', 'locker', 'tractor', 'stretcher', 'frame', 'grating', 'alembic', 'candle', 'barrier', 'cardboard', 'cave', 'puddle', 'tarp', 'price tag', 'watchtower', 'meters', 'light bulb', 'tracks', 'hair dryer', 'skirt', 'viaduct', 'paper towel', 'coat', 'sheet', 'fire extinguisher', 'water wheel', 'pottery', 'magazine rack', 'teapot', 'microphone', 'support', 'forklift', 'canyon', 'cash register', 'leaf', 'remote control', 'soap dish', 'windshield', 'cat', 'cue', 'vent', 'videos', 'shovel', 'eaves', 'antenna', 'shipyard', 'hen', 'traffic cone', 'washing machines', 'truck crane', 'cds', 'niche', 'scoreboard', 'briefcase', 'boot', 'sweater', 'hay', 'pack', 'bottle rack', 'glacier', 'pergola', 'building materials', 'television camera', 'first floor', 'rifle', 'tennis table', 'stadium', 'safety belt', 'cover', 'dish rack', 'synthesizer', 'pumpkin', 'gutter', 'fruit stand', 'ice floe', 'handle', 'wheelchair', 'mousepad', 'diploma', 'fairground ride', 'radio', 'hotplate', 'junk', 'wheelbarrow', 'stream', 'toll plaza', 'punching bag', 'trough', 'throne', 'chair desk', 'weighbridge', 'extractor fan', 'hanging clothes', 'dish', 'alarm clock', 'ski lift', 'chain', 'garage', 'mechanical shovel', 'wine rack', 'tramway', 'treadmill', 'menu', 'block', 'well', 'witness stand', 'branch', 'duck', 'casserole', 'frying pan', 'desk organizer', 'mast', 'spectacles', 'service elevator', 'dollhouse', 'hammock', 'clothes hanging', 'photocopier', 'notepad', 'golf cart', 'footpath', 'cross', 'baptismal font', 'boiler', 'skip', 'rotisserie', 'tables', 'water mill', 'helmet', 'cover curtain', 'brick', 'table runner', 'ashtray', 'street box', 'stick', 'hangers', 'cells', 'urinal', 'centerpiece', 'portable fridge', 'dvds', 'golf club', 'skirting board', 'water cooler', 'clipboard', 'camera', 'pigeonhole', 'chips', 'food processor', 'post box', 'lid', 'drum', 'blender', 'cave entrance', 'dental chair', 'obelisk', 'canoe', 'mobile', 'monitors', 'pool ball', 'cue rack', 'baggage carts', 'shore', 'fork', 'paper filer', 'bicycle rack', 'coat rack', 'garland', 'sports bag', 'fish tank', 'towel dispenser', 'carriage', 'brochure', 'plaque', 'stringer', 'iron', 'spoon', 'flag pole', 'toilet brush', 'book stand', 'water faucet', 'ticket office', 'broom', 'dvd', 'ice bucket', 'carapace', 'tureen', 'folders', 'chess', 'root', 'sewing machine', 'model', 'pen', 'violin', 'sweatshirt', 'recycling materials', 'mitten', 'chopping board', 'mask', 'log', 'mouse', 'grill', 'hole', 'target', 'trash bag', 'chalk', 'sticks', 'balloon', 'score', 'hair spray', 'roll', 'runner', 'engine', 'inflatable glove', 'games', 'pallets', 'baskets', 'coop', 'dvd player', 'rocking horse', 'buckets', 'bread rolls', 'shawl', 'watering can', 'spotlights', 'post-it', 'bowls', 'security camera', 'runner cloth', 'lock', 'alarm', 'side', 'roulette', 'bone', 'cutlery', 'pool balls', 'wheels', 'spice rack', 'plant pots', 'towel ring', 'bread box', 'video', 'funfair', 'breads', 'tripod', 'ironing board', 'skimmer', 'hollow', 'scratching post', 'tricycle', 'file box', 'mountain pass', 'tombstones', 'cooker', 'card game', 'golf bag', 'towel paper', 'chaise lounge', 'sun', 'toilet paper holder', 'rake', 'key', 'umbrella stand', 'dartboard', 'transformer', 'fireplace utensils', 'sweatshirts', 'cellular telephone', 'tallboy', 'stapler', 'sauna', 'test tube', 'palette', 'shopping carts', 'tools', 'push button', 'star', 'roof rack', 'barbed wire', 'spray', 'ear', 'sponge', 'racket', 'tins', 'eyeglasses', 'file', 'scarfs', 'sugar bowl', 'flip flop', 'headstones', 'laptop bag', 'leash', 'climbing frame', 'suit hanger', 'floor spotlight', 'plate rack', 'sewer', 'hard drive', 'sprinkler', 'tools box', 'necklace', 'bulbs', 'steel industry', 'club', 'jack', 'door bars', 'control panel', 'hairbrush', 'napkin holder', 'office', 'smoke detector', 'utensils', 'apron', 'scissors', 'terminal', 'grinder', 'entry phone', 'newspaper stand', 'pepper shaker', 'onions', 'central processing unit', 'tape', 'bat', 'coaster', 'calculator', 'potatoes', 'luggage rack', 'salt', 'street number', 'viewpoint', 'sword', 'cd', 'rowing machine', 'plug', 'andiron', 'pepper', 'tongs', 'bonfire', 'dog dish', 'belt', 'dumbbells', 'videocassette recorder', 'hook', 'envelopes', 'shower faucet', 'watch', 'padlock', 'swimming pool ladder', 'spanners', 'gravy boat', 'notice board', 'trash bags', 'fire alarm', 'ladle', 'stethoscope', 'rocket', 'funnel', 'bowling pins', 'valve', 'thermometer', 'cups', 'spice jar', 'night light', 'soaps', 'games table', 'slotted spoon', 'reel', 'scourer', 'sleeping robe', 'desk mat', 'dumbbell', 'hammer', 'tie', 'typewriter', 'shaker', 'cheese dish', 'sea star', 'racquet', 'butane gas cylinder', 'paper weight', 'shaving brush', 'sunglasses', 'gear shift', 'towel rail', 'adding machine']

LVIS_CATEGORIES = ['aerosol_can', 'air_conditioner', 'airplane', 'alarm_clock', 'alcohol', 'alligator', 'almond', 'ambulance', 'amplifier', 'anklet', 'antenna', 'apple', 'applesauce', 'apricot', 'apron', 'aquarium', 'arctic_(type_of_shoe)', 'armband', 'armchair', 'armoire', 'armor', 'artichoke', 'trash_can', 'ashtray', 'asparagus', 'atomizer', 'avocado', 'award', 'awning', 'ax', 'baboon', 'baby_buggy', 'basketball_backboard', 'backpack', 'handbag', 'suitcase', 'bagel', 'bagpipe', 'baguet', 'bait', 'ball', 'ballet_skirt', 'balloon', 'bamboo', 'banana', 'Band_Aid', 'bandage', 'bandanna', 'banjo', 'banner', 'barbell', 'barge', 'barrel', 'barrette', 'barrow', 'baseball_base', 'baseball', 'baseball_bat', 'baseball_cap', 'baseball_glove', 'basket', 'basketball', 'bass_horn', 'bat_(animal)', 'bath_mat', 'bath_towel', 'bathrobe', 'bathtub', 'batter_(food)', 'battery', 'beachball', 'bead', 'bean_curd', 'beanbag', 'beanie', 'bear', 'bed', 'bedpan', 'bedspread', 'cow', 'beef_(food)', 'beeper', 'beer_bottle', 'beer_can', 'beetle', 'bell', 'bell_pepper', 'belt', 'belt_buckle', 'bench', 'beret', 'bib', 'Bible', 'bicycle', 'visor', 'billboard', 'binder', 'binoculars', 'bird', 'birdfeeder', 'birdbath', 'birdcage', 'birdhouse', 'birthday_cake', 'birthday_card', 'pirate_flag', 'black_sheep', 'blackberry', 'blackboard', 'blanket', 'blazer', 'blender', 'blimp', 'blinker', 'blouse', 'blueberry', 'gameboard', 'boat', 'bob', 'bobbin', 'bobby_pin', 'boiled_egg', 'bolo_tie', 'deadbolt', 'bolt', 'bonnet', 'book', 'bookcase', 'booklet', 'bookmark', 'boom_microphone', 'boot', 'bottle', 'bottle_opener', 'bouquet', 'bow_(weapon)', 'bow_(decorative_ribbons)', 'bow-tie', 'bowl', 'pipe_bowl', 'bowler_hat', 'bowling_ball', 'box', 'boxing_glove', 'suspenders', 'bracelet', 'brass_plaque', 'brassiere', 'bread-bin', 'bread', 'breechcloth', 'bridal_gown', 'briefcase', 'broccoli', 'broach', 'broom', 'brownie', 'brussels_sprouts', 'bubble_gum', 'bucket', 'horse_buggy', 'horned_cow', 'bulldog', 'bulldozer', 'bullet_train', 'bulletin_board', 'bulletproof_vest', 'bullhorn', 'bun', 'bunk_bed', 'buoy', 'burrito', 'bus_(vehicle)', 'business_card', 'butter', 'butterfly', 'button', 'cab_(taxi)', 'cabana', 'cabin_car', 'cabinet', 'locker', 'cake', 'calculator', 'calendar', 'calf', 'camcorder', 'camel', 'camera', 'camera_lens', 'camper_(vehicle)', 'can', 'can_opener', 'candle', 'candle_holder', 'candy_bar', 'candy_cane', 'walking_cane', 'canister', 'canoe', 'cantaloup', 'canteen', 'cap_(headwear)', 'bottle_cap', 'cape', 'cappuccino', 'car_(automobile)', 'railcar_(part_of_a_train)', 'elevator_car', 'car_battery', 'identity_card', 'card', 'cardigan', 'cargo_ship', 'carnation', 'horse_carriage', 'carrot', 'tote_bag', 'cart', 'carton', 'cash_register', 'casserole', 'cassette', 'cast', 'cat', 'cauliflower', 'cayenne_(spice)', 'CD_player', 'celery', 'cellular_telephone', 'chain_mail', 'chair', 'chaise_longue', 'chalice', 'chandelier', 'chap', 'checkbook', 'checkerboard', 'cherry', 'chessboard', 'chicken_(animal)', 'chickpea', 'chili_(vegetable)', 'chime', 'chinaware', 'crisp_(potato_chip)', 'poker_chip', 'chocolate_bar', 'chocolate_cake', 'chocolate_milk', 'chocolate_mousse', 'choker', 'chopping_board', 'chopstick', 'Christmas_tree', 'slide', 'cider', 'cigar_box', 'cigarette', 'cigarette_case', 'cistern', 'clarinet', 'clasp', 'cleansing_agent', 'cleat_(for_securing_rope)', 'clementine', 'clip', 'clipboard', 'clippers_(for_plants)', 'cloak', 'clock', 'clock_tower', 'clothes_hamper', 'clothespin', 'clutch_bag', 'coaster', 'coat', 'coat_hanger', 'coatrack', 'cock', 'cockroach', 'cocoa_(beverage)', 'coconut', 'coffee_maker', 'coffee_table', 'coffeepot', 'coil', 'coin', 'colander', 'coleslaw', 'coloring_material', 'combination_lock', 'pacifier', 'comic_book', 'compass', 'computer_keyboard', 'condiment', 'cone', 'control', 'convertible_(automobile)', 'sofa_bed', 'cooker', 'cookie', 'cooking_utensil', 'cooler_(for_food)', 'cork_(bottle_plug)', 'corkboard', 'corkscrew', 'edible_corn', 'cornbread', 'cornet', 'cornice', 'cornmeal', 'corset', 'costume', 'cougar', 'coverall', 'cowbell', 'cowboy_hat', 'crab_(animal)', 'crabmeat', 'cracker', 'crape', 'crate', 'crayon', 'cream_pitcher', 'crescent_roll', 'crib', 'crock_pot', 'crossbar', 'crouton', 'crow', 'crowbar', 'crown', 'crucifix', 'cruise_ship', 'police_cruiser', 'crumb', 'crutch', 'cub_(animal)', 'cube', 'cucumber', 'cufflink', 'cup', 'trophy_cup', 'cupboard', 'cupcake', 'hair_curler', 'curling_iron', 'curtain', 'cushion', 'cylinder', 'cymbal', 'dagger', 'dalmatian', 'dartboard', 'date_(fruit)', 'deck_chair', 'deer', 'dental_floss', 'desk', 'detergent', 'diaper', 'diary', 'die', 'dinghy', 'dining_table', 'tux', 'dish', 'dish_antenna', 'dishrag', 'dishtowel', 'dishwasher', 'dishwasher_detergent', 'dispenser', 'diving_board', 'Dixie_cup', 'dog', 'dog_collar', 'doll', 'dollar', 'dollhouse', 'dolphin', 'domestic_ass', 'doorknob', 'doormat', 'doughnut', 'dove', 'dragonfly', 'drawer', 'underdrawers', 'dress', 'dress_hat', 'dress_suit', 'dresser', 'drill', 'drone', 'dropper', 'drum_(musical_instrument)', 'drumstick', 'duck', 'duckling', 'duct_tape', 'duffel_bag', 'dumbbell', 'dumpster', 'dustpan', 'eagle', 'earphone', 'earplug', 'earring', 'easel', 'eclair', 'eel', 'egg', 'egg_roll', 'egg_yolk', 'eggbeater', 'eggplant', 'electric_chair', 'refrigerator', 'elephant', 'elk', 'envelope', 'eraser', 'escargot', 'eyepatch', 'falcon', 'fan', 'faucet', 'fedora', 'ferret', 'Ferris_wheel', 'ferry', 'fig_(fruit)', 'fighter_jet', 'figurine', 'file_cabinet', 'file_(tool)', 'fire_alarm', 'fire_engine', 'fire_extinguisher', 'fire_hose', 'fireplace', 'fireplug', 'first-aid_kit', 'fish', 'fish_(food)', 'fishbowl', 'fishing_rod', 'flag', 'flagpole', 'flamingo', 'flannel', 'flap', 'flash', 'flashlight', 'fleece', 'flip-flop_(sandal)', 'flipper_(footwear)', 'flower_arrangement', 'flute_glass', 'foal', 'folding_chair', 'food_processor', 'football_(American)', 'football_helmet', 'footstool', 'fork', 'forklift', 'freight_car', 'French_toast', 'freshener', 'frisbee', 'frog', 'fruit_juice', 'frying_pan', 'fudge', 'funnel', 'futon', 'gag', 'garbage', 'garbage_truck', 'garden_hose', 'gargle', 'gargoyle', 'garlic', 'gasmask', 'gazelle', 'gelatin', 'gemstone', 'generator', 'giant_panda', 'gift_wrap', 'ginger', 'giraffe', 'cincture', 'glass_(drink_container)', 'globe', 'glove', 'goat', 'goggles', 'goldfish', 'golf_club', 'golfcart', 'gondola_(boat)', 'goose', 'gorilla', 'gourd', 'grape', 'grater', 'gravestone', 'gravy_boat', 'green_bean', 'green_onion', 'griddle', 'grill', 'grits', 'grizzly', 'grocery_bag', 'guitar', 'gull', 'gun', 'hairbrush', 'hairnet', 'hairpin', 'halter_top', 'ham', 'hamburger', 'hammer', 'hammock', 'hamper', 'hamster', 'hair_dryer', 'hand_glass', 'hand_towel', 'handcart', 'handcuff', 'handkerchief', 'handle', 'handsaw', 'hardback_book', 'harmonium', 'hat', 'hatbox', 'veil', 'headband', 'headboard', 'headlight', 'headscarf', 'headset', 'headstall_(for_horses)', 'heart', 'heater', 'helicopter', 'helmet', 'heron', 'highchair', 'hinge', 'hippopotamus', 'hockey_stick', 'hog', 'home_plate_(baseball)', 'honey', 'fume_hood', 'hook', 'hookah', 'hornet', 'horse', 'hose', 'hot-air_balloon', 'hotplate', 'hot_sauce', 'hourglass', 'houseboat', 'hummingbird', 'hummus', 'polar_bear', 'icecream', 'popsicle', 'ice_maker', 'ice_pack', 'ice_skate', 'igniter', 'inhaler', 'iPod', 'iron_(for_clothing)', 'ironing_board', 'jacket', 'jam', 'jar', 'jean', 'jeep', 'jelly_bean', 'jersey', 'jet_plane', 'jewel', 'jewelry', 'joystick', 'jumpsuit', 'kayak', 'keg', 'kennel', 'kettle', 'key', 'keycard', 'kilt', 'kimono', 'kitchen_sink', 'kitchen_table', 'kite', 'kitten', 'kiwi_fruit', 'knee_pad', 'knife', 'knitting_needle', 'knob', 'knocker_(on_a_door)', 'koala', 'lab_coat', 'ladder', 'ladle', 'ladybug', 'lamb_(animal)', 'lamb-chop', 'lamp', 'lamppost', 'lampshade', 'lantern', 'lanyard', 'laptop_computer', 'lasagna', 'latch', 'lawn_mower', 'leather', 'legging_(clothing)', 'Lego', 'legume', 'lemon', 'lemonade', 'lettuce', 'license_plate', 'life_buoy', 'life_jacket', 'lightbulb', 'lightning_rod', 'lime', 'limousine', 'lion', 'lip_balm', 'liquor', 'lizard', 'log', 'lollipop', 'speaker_(stero_equipment)', 'loveseat', 'machine_gun', 'magazine', 'magnet', 'mail_slot', 'mailbox_(at_home)', 'mallard', 'mallet', 'mammoth', 'manatee', 'mandarin_orange', 'manger', 'manhole', 'map', 'marker', 'martini', 'mascot', 'mashed_potato', 'masher', 'mask', 'mast', 'mat_(gym_equipment)', 'matchbox', 'mattress', 'measuring_cup', 'measuring_stick', 'meatball', 'medicine', 'melon', 'microphone', 'microscope', 'microwave_oven', 'milestone', 'milk', 'milk_can', 'milkshake', 'minivan', 'mint_candy', 'mirror', 'mitten', 'mixer_(kitchen_tool)', 'money', 'monitor_(computer_equipment) computer_monitor', 'monkey', 'motor', 'motor_scooter', 'motor_vehicle', 'motorcycle', 'mound_(baseball)', 'mouse_(computer_equipment)', 'mousepad', 'muffin', 'mug', 'mushroom', 'music_stool', 'musical_instrument', 'nailfile', 'napkin', 'neckerchief', 'necklace', 'necktie', 'needle', 'nest', 'newspaper', 'newsstand', 'nightshirt', 'nosebag_(for_animals)', 'noseband_(for_animals)', 'notebook', 'notepad', 'nut', 'nutcracker', 'oar', 'octopus_(food)', 'octopus_(animal)', 'oil_lamp', 'olive_oil', 'omelet', 'onion', 'orange_(fruit)', 'orange_juice', 'ostrich', 'ottoman', 'oven', 'overalls_(clothing)', 'owl', 'packet', 'inkpad', 'pad', 'paddle', 'padlock', 'paintbrush', 'painting', 'pajamas', 'palette', 'pan_(for_cooking)', 'pan_(metal_container)', 'pancake', 'pantyhose', 'papaya', 'paper_plate', 'paper_towel', 'paperback_book', 'paperweight', 'parachute', 'parakeet', 'parasail_(sports)', 'parasol', 'parchment', 'parka', 'parking_meter', 'parrot', 'passenger_car_(part_of_a_train)', 'passenger_ship', 'passport', 'pastry', 'patty_(food)', 'pea_(food)', 'peach', 'peanut_butter', 'pear', 'peeler_(tool_for_fruit_and_vegetables)', 'wooden_leg', 'pegboard', 'pelican', 'pen', 'pencil', 'pencil_box', 'pencil_sharpener', 'pendulum', 'penguin', 'pennant', 'penny_(coin)', 'pepper', 'pepper_mill', 'perfume', 'persimmon', 'person', 'pet', 'pew_(church_bench)', 'phonebook', 'phonograph_record', 'piano', 'pickle', 'pickup_truck', 'pie', 'pigeon', 'piggy_bank', 'pillow', 'pin_(non_jewelry)', 'pineapple', 'pinecone', 'ping-pong_ball', 'pinwheel', 'tobacco_pipe', 'pipe', 'pistol', 'pita_(bread)', 'pitcher_(vessel_for_liquid)', 'pitchfork', 'pizza', 'place_mat', 'plate', 'platter', 'playpen', 'pliers', 'plow_(farm_equipment)', 'plume', 'pocket_watch', 'pocketknife', 'poker_(fire_stirring_tool)', 'pole', 'polo_shirt', 'poncho', 'pony', 'pool_table', 'pop_(soda)', 'postbox_(public)', 'postcard', 'poster', 'pot', 'flowerpot', 'potato', 'potholder', 'pottery', 'pouch', 'power_shovel', 'prawn', 'pretzel', 'printer', 'projectile_(weapon)', 'projector', 'propeller', 'prune', 'pudding', 'puffer_(fish)', 'puffin', 'pug-dog', 'pumpkin', 'puncher', 'puppet', 'puppy', 'quesadilla', 'quiche', 'quilt', 'rabbit', 'race_car', 'racket', 'radar', 'radiator', 'radio_receiver', 'radish', 'raft', 'rag_doll', 'raincoat', 'ram_(animal)', 'raspberry', 'rat', 'razorblade', 'reamer_(juicer)', 'rearview_mirror', 'receipt', 'recliner', 'record_player', 'reflector', 'remote_control', 'rhinoceros', 'rib_(food)', 'rifle', 'ring', 'river_boat', 'road_map', 'robe', 'rocking_chair', 'rodent', 'roller_skate', 'Rollerblade', 'rolling_pin', 'root_beer', 'router_(computer_equipment)', 'rubber_band', 'runner_(carpet)', 'plastic_bag', 'saddle_(on_an_animal)', 'saddle_blanket', 'saddlebag', 'safety_pin', 'sail', 'salad', 'salad_plate', 'salami', 'salmon_(fish)', 'salmon_(food)', 'salsa', 'saltshaker', 'sandal_(type_of_shoe)', 'sandwich', 'satchel', 'saucepan', 'saucer', 'sausage', 'sawhorse', 'saxophone', 'scale_(measuring_instrument)', 'scarecrow', 'scarf', 'school_bus', 'scissors', 'scoreboard', 'scraper', 'screwdriver', 'scrubbing_brush', 'sculpture', 'seabird', 'seahorse', 'seaplane', 'seashell', 'sewing_machine', 'shaker', 'shampoo', 'shark', 'sharpener', 'Sharpie', 'shaver_(electric)', 'shaving_cream', 'shawl', 'shears', 'sheep', 'shepherd_dog', 'sherbert', 'shield', 'shirt', 'shoe', 'shopping_bag', 'shopping_cart', 'short_pants', 'shot_glass', 'shoulder_bag', 'shovel', 'shower_head', 'shower_cap', 'shower_curtain', 'shredder_(for_paper)', 'signboard', 'silo', 'sink', 'skateboard', 'skewer', 'ski', 'ski_boot', 'ski_parka', 'ski_pole', 'skirt', 'skullcap', 'sled', 'sleeping_bag', 'sling_(bandage)', 'slipper_(footwear)', 'smoothie', 'snake', 'snowboard', 'snowman', 'snowmobile', 'soap', 'soccer_ball', 'sock', 'sofa', 'softball', 'solar_array', 'sombrero', 'soup', 'soup_bowl', 'soupspoon', 'sour_cream', 'soya_milk', 'space_shuttle', 'sparkler_(fireworks)', 'spatula', 'spear', 'spectacles', 'spice_rack', 'spider', 'crawfish', 'sponge', 'spoon', 'sportswear', 'spotlight', 'squid_(food)', 'squirrel', 'stagecoach', 'stapler_(stapling_machine)', 'starfish', 'statue_(sculpture)', 'steak_(food)', 'steak_knife', 'steering_wheel', 'stepladder', 'step_stool', 'stereo_(sound_system)', 'stew', 'stirrer', 'stirrup', 'stool', 'stop_sign', 'brake_light', 'stove', 'strainer', 'strap', 'straw_(for_drinking)', 'strawberry', 'street_sign', 'streetlight', 'string_cheese', 'stylus', 'subwoofer', 'sugar_bowl', 'sugarcane_(plant)', 'suit_(clothing)', 'sunflower', 'sunglasses', 'sunhat', 'surfboard', 'sushi', 'mop', 'sweat_pants', 'sweatband', 'sweater', 'sweatshirt', 'sweet_potato', 'swimsuit', 'sword', 'syringe', 'Tabasco_sauce', 'table-tennis_table', 'table', 'table_lamp', 'tablecloth', 'tachometer', 'taco', 'tag', 'taillight', 'tambourine', 'army_tank', 'tank_(storage_vessel)', 'tank_top_(clothing)', 'tape_(sticky_cloth_or_paper)', 'tape_measure', 'tapestry', 'tarp', 'tartan', 'tassel', 'tea_bag', 'teacup', 'teakettle', 'teapot', 'teddy_bear', 'telephone', 'telephone_booth', 'telephone_pole', 'telephoto_lens', 'television_camera', 'television_set', 'tennis_ball', 'tennis_racket', 'tequila', 'thermometer', 'thermos_bottle', 'thermostat', 'thimble', 'thread', 'thumbtack', 'tiara', 'tiger', 'tights_(clothing)', 'timer', 'tinfoil', 'tinsel', 'tissue_paper', 'toast_(food)', 'toaster', 'toaster_oven', 'toilet', 'toilet_tissue', 'tomato', 'tongs', 'toolbox', 'toothbrush', 'toothpaste', 'toothpick', 'cover', 'tortilla', 'tow_truck', 'towel', 'towel_rack', 'toy', 'tractor_(farm_equipment)', 'traffic_light', 'dirt_bike', 'trailer_truck', 'train_(railroad_vehicle)', 'trampoline', 'tray', 'trench_coat', 'triangle_(musical_instrument)', 'tricycle', 'tripod', 'trousers', 'truck', 'truffle_(chocolate)', 'trunk', 'vat', 'turban', 'turkey_(food)', 'turnip', 'turtle', 'turtleneck_(clothing)', 'typewriter', 'umbrella', 'underwear', 'unicycle', 'urinal', 'urn', 'vacuum_cleaner', 'vase', 'vending_machine', 'vent', 'vest', 'videotape', 'vinegar', 'violin', 'vodka', 'volleyball', 'vulture', 'waffle', 'waffle_iron', 'wagon', 'wagon_wheel', 'walking_stick', 'wall_clock', 'wall_socket', 'wallet', 'walrus', 'wardrobe', 'washbasin', 'automatic_washer', 'watch', 'water_bottle', 'water_cooler', 'water_faucet', 'water_heater', 'water_jug', 'water_gun', 'water_scooter', 'water_ski', 'water_tower', 'watering_can', 'watermelon', 'weathervane', 'webcam', 'wedding_cake', 'wedding_ring', 'wet_suit', 'wheel', 'wheelchair', 'whipped_cream', 'whistle', 'wig', 'wind_chime', 'windmill', 'window_box_(for_plants)', 'windshield_wiper', 'windsock', 'wine_bottle', 'wine_bucket', 'wineglass', 'blinder_(for_horses)', 'wok', 'wolf', 'wooden_spoon', 'wreath', 'wrench', 'wristband', 'wristlet', 'yacht', 'yogurt', 'yoke_(animal_equipment)', 'zebra', 'zucchini']


objects365_categories = [
{'id': 164, 'name': 'cutting/chopping board'} ,
{'id': 49, 'name': 'tie'} ,
{'id': 306, 'name': 'crosswalk sign'} ,
{'id': 145, 'name': 'gun'} ,
{'id': 14, 'name': 'street lights'} ,
{'id': 223, 'name': 'bar soap'} ,
{'id': 74, 'name': 'wild bird'} ,
{'id': 219, 'name': 'ice cream'} ,
{'id': 37, 'name': 'stool'} ,
{'id': 25, 'name': 'storage box'} ,
{'id': 153, 'name': 'giraffe'} ,
{'id': 52, 'name': 'pen/pencil'} ,
{'id': 61, 'name': 'high heels'} ,
{'id': 340, 'name': 'mangosteen'} ,
{'id': 22, 'name': 'bracelet'} ,
{'id': 155, 'name': 'piano'} ,
{'id': 162, 'name': 'vent'} ,
{'id': 75, 'name': 'laptop'} ,
{'id': 236, 'name': 'toaster'} ,
{'id': 231, 'name': 'fire truck'} ,
{'id': 42, 'name': 'basket'} ,
{'id': 150, 'name': 'zebra'} ,
{'id': 124, 'name': 'head phone'} ,
{'id': 90, 'name': 'sheep'} ,
{'id': 322, 'name': 'steak'} ,
{'id': 39, 'name': 'couch'} ,
{'id': 209, 'name': 'toothbrush'} ,
{'id': 59, 'name': 'bicycle'} ,
{'id': 336, 'name': 'red cabbage'} ,
{'id': 228, 'name': 'golf ball'} ,
{'id': 120, 'name': 'tomato'} ,
{'id': 132, 'name': 'computer box'} ,
{'id': 8, 'name': 'cup'} ,
{'id': 183, 'name': 'basketball'} ,
{'id': 298, 'name': 'butterfly'} ,
{'id': 250, 'name': 'garlic'} ,
{'id': 12, 'name': 'desk'} ,
{'id': 141, 'name': 'microwave'} ,
{'id': 171, 'name': 'strawberry'} ,
{'id': 200, 'name': 'kettle'} ,
{'id': 63, 'name': 'van'} ,
{'id': 300, 'name': 'cheese'} ,
{'id': 215, 'name': 'marker'} ,
{'id': 100, 'name': 'blackboard/whiteboard'} ,
{'id': 186, 'name': 'printer'} ,
{'id': 333, 'name': 'bread/bun'} ,
{'id': 243, 'name': 'penguin'} ,
{'id': 364, 'name': 'iron'} ,
{'id': 180, 'name': 'ladder'} ,
{'id': 34, 'name': 'flag'} ,
{'id': 78, 'name': 'cell phone'} ,
{'id': 97, 'name': 'fan'} ,
{'id': 224, 'name': 'scale'} ,
{'id': 151, 'name': 'duck'} ,
{'id': 319, 'name': 'flute'} ,
{'id': 156, 'name': 'stop sign'} ,
{'id': 290, 'name': 'rickshaw'} ,
{'id': 128, 'name': 'sailboat'} ,
{'id': 165, 'name': 'tennis racket'} ,
{'id': 241, 'name': 'cigar'} ,
{'id': 101, 'name': 'balloon'} ,
{'id': 308, 'name': 'hair drier'} ,
{'id': 167, 'name': 'skating and skiing shoes'} ,
{'id': 237, 'name': 'helicopter'} ,
{'id': 65, 'name': 'sink'} ,
{'id': 129, 'name': 'tangerine'} ,
{'id': 330, 'name': 'crab'} ,
{'id': 320, 'name': 'measuring cup'} ,
{'id': 260, 'name': 'fishing rod'} ,
{'id': 346, 'name': 'saw'} ,
{'id': 216, 'name': 'ship'} ,
{'id': 46, 'name': 'coffee table'} ,
{'id': 194, 'name': 'facial mask'} ,
{'id': 281, 'name': 'stapler'} ,
{'id': 118, 'name': 'refrigerator'} ,
{'id': 40, 'name': 'belt'} ,
{'id': 349, 'name': 'starfish'} ,
{'id': 87, 'name': 'hanger'} ,
{'id': 116, 'name': 'baseball glove'} ,
{'id': 261, 'name': 'cherry'} ,
{'id': 334, 'name': 'baozi'} ,
{'id': 267, 'name': 'screwdriver'} ,
{'id': 158, 'name': 'converter'} ,
{'id': 335, 'name': 'lion'} ,
{'id': 170, 'name': 'baseball'} ,
{'id': 111, 'name': 'skis'} ,
{'id': 136, 'name': 'broccoli'} ,
{'id': 342, 'name': 'eraser'} ,
{'id': 337, 'name': 'polar bear'} ,
{'id': 139, 'name': 'shovel'} ,
{'id': 193, 'name': 'extension cord'} ,
{'id': 284, 'name': 'goldfish'} ,
{'id': 174, 'name': 'pepper'} ,
{'id': 138, 'name': 'stroller'} ,
{'id': 328, 'name': 'yak'} ,
{'id': 83, 'name': 'clock'} ,
{'id': 235, 'name': 'tricycle'} ,
{'id': 248, 'name': 'parking meter'} ,
{'id': 274, 'name': 'trophy'} ,
{'id': 324, 'name': 'binoculars'} ,
{'id': 51, 'name': 'traffic light'} ,
{'id': 314, 'name': 'donkey'} ,
{'id': 45, 'name': 'barrel/bucket'} ,
{'id': 292, 'name': 'pomegranate'} ,
{'id': 13, 'name': 'handbag'} ,
{'id': 262, 'name': 'tablet'} ,
{'id': 68, 'name': 'apple'} ,
{'id': 226, 'name': 'cabbage'} ,
{'id': 23, 'name': 'flower'} ,
{'id': 58, 'name': 'faucet'} ,
{'id': 206, 'name': 'tong'} ,
{'id': 291, 'name': 'trombone'} ,
{'id': 160, 'name': 'carrot'} ,
{'id': 172, 'name': 'bow tie'} ,
{'id': 122, 'name': 'tent'} ,
{'id': 163, 'name': 'cookies'} ,
{'id': 115, 'name': 'remote'} ,
{'id': 175, 'name': 'coffee machine'} ,
{'id': 238, 'name': 'green beans'} ,
{'id': 233, 'name': 'cello'} ,
{'id': 28, 'name': 'wine glass'} ,
{'id': 295, 'name': 'mushroom'} ,
{'id': 344, 'name': 'scallop'} ,
{'id': 125, 'name': 'lantern'} ,
{'id': 123, 'name': 'shampoo/shower gel'} ,
{'id': 285, 'name': 'meat balls'} ,
{'id': 266, 'name': 'key'} ,
{'id': 296, 'name': 'calculator'} ,
{'id': 168, 'name': 'scissors'} ,
{'id': 103, 'name': 'cymbal'} ,
{'id': 6, 'name': 'bottle'} ,
{'id': 264, 'name': 'nuts'} ,
{'id': 234, 'name': 'notepaper'} ,
{'id': 211, 'name': 'mango'} ,
{'id': 287, 'name': 'toothpaste'} ,
{'id': 196, 'name': 'chopsticks'} ,
{'id': 140, 'name': 'baseball bat'} ,
{'id': 244, 'name': 'hurdle'} ,
{'id': 195, 'name': 'tennis ball'} ,
{'id': 144, 'name': 'surveillance camera'} ,
{'id': 271, 'name': 'volleyball'} ,
{'id': 94, 'name': 'keyboard'} ,
{'id': 339, 'name': 'seal'} ,
{'id': 11, 'name': 'picture/frame'} ,
{'id': 348, 'name': 'okra'} ,
{'id': 191, 'name': 'sausage'} ,
{'id': 166, 'name': 'candy'} ,
{'id': 62, 'name': 'ring'} ,
{'id': 311, 'name': 'dolphin'} ,
{'id': 273, 'name': 'eggplant'} ,
{'id': 84, 'name': 'drum'} ,
{'id': 143, 'name': 'surfboard'} ,
{'id': 288, 'name': 'antelope'} ,
{'id': 204, 'name': 'clutch'} ,
{'id': 207, 'name': 'slide'} ,
{'id': 43, 'name': 'towel/napkin'} ,
{'id': 352, 'name': 'durian'} ,
{'id': 276, 'name': 'board eraser'} ,
{'id': 315, 'name': 'electric drill'} ,
{'id': 312, 'name': 'sushi'} ,
{'id': 198, 'name': 'pie'} ,
{'id': 106, 'name': 'pickup truck'} ,
{'id': 176, 'name': 'bathtub'} ,
{'id': 26, 'name': 'vase'} ,
{'id': 133, 'name': 'elephant'} ,
{'id': 256, 'name': 'sandwich'} ,
{'id': 327, 'name': 'noodles'} ,
{'id': 10, 'name': 'glasses'} ,
{'id': 109, 'name': 'airplane'} ,
{'id': 95, 'name': 'tripod'} ,
{'id': 247, 'name': 'CD'} ,
{'id': 121, 'name': 'machinery vehicle'} ,
{'id': 365, 'name': 'flashlight'} ,
{'id': 53, 'name': 'microphone'} ,
{'id': 270, 'name': 'pliers'} ,
{'id': 362, 'name': 'chainsaw'} ,
{'id': 259, 'name': 'bear'} ,
{'id': 197, 'name': 'electronic stove and gas stove'} ,
{'id': 89, 'name': 'pot/pan'} ,
{'id': 220, 'name': 'tape'} ,
{'id': 338, 'name': 'lighter'} ,
{'id': 177, 'name': 'snowboard'} ,
{'id': 214, 'name': 'violin'} ,
{'id': 217, 'name': 'chicken'} ,
{'id': 2, 'name': 'sneakers'} ,
{'id': 161, 'name': 'washing machine'} ,
{'id': 131, 'name': 'kite'} ,
{'id': 354, 'name': 'rabbit'} ,
{'id': 86, 'name': 'bus'} ,
{'id': 275, 'name': 'dates'} ,
{'id': 282, 'name': 'camel'} ,
{'id': 88, 'name': 'nightstand'} ,
{'id': 179, 'name': 'grapes'} ,
{'id': 229, 'name': 'pine apple'} ,
{'id': 56, 'name': 'necklace'} ,
{'id': 18, 'name': 'leather shoes'} ,
{'id': 358, 'name': 'hoverboard'} ,
{'id': 345, 'name': 'pencil case'} ,
{'id': 359, 'name': 'pasta'} ,
{'id': 157, 'name': 'radiator'} ,
{'id': 201, 'name': 'hamburger'} ,
{'id': 268, 'name': 'globe'} ,
{'id': 332, 'name': 'barbell'} ,
{'id': 329, 'name': 'mop'} ,
{'id': 252, 'name': 'horn'} ,
{'id': 350, 'name': 'eagle'} ,
{'id': 169, 'name': 'folder'} ,
{'id': 137, 'name': 'toilet'} ,
{'id': 5, 'name': 'lamp'} ,
{'id': 27, 'name': 'bench'} ,
{'id': 249, 'name': 'swan'} ,
{'id': 76, 'name': 'knife'} ,
{'id': 341, 'name': 'comb'} ,
{'id': 64, 'name': 'watch'} ,
{'id': 105, 'name': 'telephone'} ,
{'id': 3, 'name': 'chair'} ,
{'id': 33, 'name': 'boat'} ,
{'id': 107, 'name': 'orange'} ,
{'id': 60, 'name': 'bread'} ,
{'id': 147, 'name': 'cat'} ,
{'id': 135, 'name': 'gas stove'} ,
{'id': 307, 'name': 'papaya'} ,
{'id': 227, 'name': 'router/modem'} ,
{'id': 357, 'name': 'asparagus'} ,
{'id': 73, 'name': 'motorcycle'} ,
{'id': 77, 'name': 'traffic sign'} ,
{'id': 67, 'name': 'fish'} ,
{'id': 326, 'name': 'radish'} ,
{'id': 213, 'name': 'egg'} ,
{'id': 203, 'name': 'cucumber'} ,
{'id': 17, 'name': 'helmet'} ,
{'id': 110, 'name': 'luggage'} ,
{'id': 80, 'name': 'truck'} ,
{'id': 199, 'name': 'frisbee'} ,
{'id': 232, 'name': 'peach'} ,
{'id': 1, 'name': 'person'} ,
{'id': 29, 'name': 'boots'} ,
{'id': 310, 'name': 'chips'} ,
{'id': 142, 'name': 'skateboard'} ,
{'id': 44, 'name': 'slippers'} ,
{'id': 4, 'name': 'hat'} ,
{'id': 178, 'name': 'suitcase'} ,
{'id': 24, 'name': 'tv'} ,
{'id': 119, 'name': 'train'} ,
{'id': 82, 'name': 'power outlet'} ,
{'id': 245, 'name': 'swing'} ,
{'id': 15, 'name': 'book'} ,
{'id': 294, 'name': 'jellyfish'} ,
{'id': 192, 'name': 'fire extinguisher'} ,
{'id': 212, 'name': 'deer'} ,
{'id': 181, 'name': 'pear'} ,
{'id': 347, 'name': 'table tennis paddle'} ,
{'id': 113, 'name': 'trolley'} ,
{'id': 91, 'name': 'guitar'} ,
{'id': 202, 'name': 'golf club'} ,
{'id': 221, 'name': 'wheelchair'} ,
{'id': 254, 'name': 'saxophone'} ,
{'id': 117, 'name': 'paper towel'} ,
{'id': 303, 'name': 'race car'} ,
{'id': 240, 'name': 'carriage'} ,
{'id': 246, 'name': 'radio'} ,
{'id': 318, 'name': 'parrot'} ,
{'id': 251, 'name': 'french fries'} ,
{'id': 98, 'name': 'dog'} ,
{'id': 112, 'name': 'soccer'} ,
{'id': 355, 'name': 'french horn'} ,
{'id': 79, 'name': 'paddle'} ,
{'id': 283, 'name': 'lettuce'} ,
{'id': 9, 'name': 'car'} ,
{'id': 258, 'name': 'kiwi fruit'} ,
{'id': 325, 'name': 'llama'} ,
{'id': 187, 'name': 'billiards'} ,
{'id': 210, 'name': 'facial cleanser'} ,
{'id': 81, 'name': 'cow'} ,
{'id': 331, 'name': 'microscope'} ,
{'id': 148, 'name': 'lemon'} ,
{'id': 302, 'name': 'pomelo'} ,
{'id': 85, 'name': 'fork'} ,
{'id': 154, 'name': 'pumpkin'} ,
{'id': 289, 'name': 'shrimp'} ,
{'id': 71, 'name': 'teddy bear'} ,
{'id': 184, 'name': 'potato'} ,
{'id': 102, 'name': 'air conditioner'} ,
{'id': 208, 'name': 'hot dog'} ,
{'id': 222, 'name': 'plum'} ,
{'id': 316, 'name': 'spring rolls'} ,
{'id': 230, 'name': 'crane'} ,
{'id': 149, 'name': 'liquid soap'} ,
{'id': 55, 'name': 'canned'} ,
{'id': 35, 'name': 'speaker'} ,
{'id': 108, 'name': 'banana'} ,
{'id': 297, 'name': 'treadmill'} ,
{'id': 99, 'name': 'spoon'} ,
{'id': 104, 'name': 'mouse'} ,
{'id': 182, 'name': 'american football'} ,
{'id': 299, 'name': 'egg tart'} ,
{'id': 127, 'name': 'cleaning products'} ,
{'id': 313, 'name': 'urinal'} ,
{'id': 286, 'name': 'medal'} ,
{'id': 239, 'name': 'brush'} ,
{'id': 96, 'name': 'hockey'} ,
{'id': 279, 'name': 'dumbbell'} ,
{'id': 32, 'name': 'umbrella'} ,
{'id': 272, 'name': 'hammer'} ,
{'id': 16, 'name': 'plate'} ,
{'id': 21, 'name': 'potted plant'} ,
{'id': 242, 'name': 'earphone'} ,
{'id': 70, 'name': 'candle'} ,
{'id': 185, 'name': 'paint brush'} ,
{'id': 48, 'name': 'toy'} ,
{'id': 130, 'name': 'pizza'} ,
{'id': 255, 'name': 'trumpet'} ,
{'id': 361, 'name': 'hotair balloon'} ,
{'id': 188, 'name': 'fire hydrant'} ,
{'id': 50, 'name': 'bed'} ,
{'id': 253, 'name': 'avocado'} ,
{'id': 293, 'name': 'coconut'} ,
{'id': 257, 'name': 'cue'} ,
{'id': 280, 'name': 'hamimelon'} ,
{'id': 66, 'name': 'horse'} ,
{'id': 173, 'name': 'pigeon'} ,
{'id': 190, 'name': 'projector'} ,
{'id': 69, 'name': 'camera'} ,
{'id': 30, 'name': 'bowl'} ,
{'id': 269, 'name': 'broom'} ,
{'id': 343, 'name': 'pitaya'} ,
{'id': 305, 'name': 'tuba'} ,
{'id': 309, 'name': 'green onion'} ,
{'id': 363, 'name': 'lobster'} ,
{'id': 225, 'name': 'watermelon'} ,
{'id': 47, 'name': 'suv'} ,
{'id': 31, 'name': 'dining table'} ,
{'id': 54, 'name': 'sandals'} ,
{'id': 351, 'name': 'monkey'} ,
{'id': 218, 'name': 'onion'} ,
{'id': 36, 'name': 'trash bin/can'} ,
{'id': 20, 'name': 'glove'} ,
{'id': 277, 'name': 'rice'} ,
{'id': 152, 'name': 'sports car'} ,
{'id': 360, 'name': 'target'} ,
{'id': 205, 'name': 'blender'} ,
{'id': 19, 'name': 'pillow'} ,
{'id': 72, 'name': 'cake'} ,
{'id': 93, 'name': 'tea pot'} ,
{'id': 353, 'name': 'game board'} ,
{'id': 38, 'name': 'backpack'} ,
{'id': 356, 'name': 'ambulance'} ,
{'id': 146, 'name': 'life saver'} ,
{'id': 189, 'name': 'goose'} ,
{'id': 278, 'name': 'tape measure/ruler'} ,
{'id': 92, 'name': 'traffic cone'} ,
{'id': 134, 'name': 'toiletries'} ,
{'id': 114, 'name': 'oven'} ,
{'id': 317, 'name': 'tortoise/turtle'} ,
{'id': 265, 'name': 'corn'} ,
{'id': 126, 'name': 'donut'} ,
{'id': 57, 'name': 'mirror'} ,
{'id': 7, 'name': 'cabinet/shelf'} ,
{'id': 263, 'name': 'green vegetables'} ,
{'id': 159, 'name': 'tissue '} ,
{'id': 321, 'name': 'shark'} ,
{'id': 301, 'name': 'pig'} ,
{'id': 41, 'name': 'carpet'} ,
{'id': 304, 'name': 'rice cooker'} ,
{'id': 323, 'name': 'poker card'} ,
]
def _get_builtin_metadata(categories):
    id_to_name = {x['id']: x['name'] for x in categories}
    thing_dataset_id_to_contiguous_id = {i + 1: i for i in range(len(categories))}
    thing_classes = [id_to_name[k] for k in sorted(id_to_name)]

    return {
        "thing_dataset_id_to_contiguous_id": thing_dataset_id_to_contiguous_id,
        "thing_classes": thing_classes}

objects365_dict = _get_builtin_metadata(objects365_categories)
OBJECTS365V1_DICT = objects365_dict["thing_dataset_id_to_contiguous_id"]
OBJECTS365V1 = objects365_dict["thing_classes"]

#####build V3DET

# from pycocotools.coco import COCO
# annFile = './datasets/v3det/v3det_2023_v1_val.json'

# coco = COCO(annFile)

# # Get all category IDs
# catIds = coco.getCatIds()

# # Load category information
# categories = coco.loadCats(catIds)

from regionspot.data.v3det_categories import categories

v3det_dict = _get_builtin_metadata(categories)
V3DET_DICT = v3det_dict["thing_dataset_id_to_contiguous_id"]
V3DET = v3det_dict["thing_classes"]


# # # #####build OpenImages
# annFile = './datasets/re_openimages_v6_train_bbox_splitdir_int_ids.json'

# openimages = COCO(annFile)

# # Get all category IDs
# openimages_catIds = openimages.getCatIds()

# # Load category information
# openimages_categories = openimages.loadCats(openimages_catIds)


# openimages_dict = _get_builtin_metadata(openimages_categories)
# openimages_DICT = openimages_dict["thing_dataset_id_to_contiguous_id"]
# openimages = openimages_dict["thing_classes"]





