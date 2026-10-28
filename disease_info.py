"""
Disease information shown AFTER the classifier predicts a class.

Only well-known, general facts are included. No pesticide / medicine names or
dosages are given on purpose - this is an academic project.
If the predicted class is not in the table, general management advice is shown.
"""
import re

DISCLAIMER = ("This is an academic project and the result should be verified by an "
              "agricultural expert before treatment.")

GENERAL_MANAGEMENT = [
    "Remove and destroy heavily affected leaves; do not compost them.",
    "Improve air circulation (spacing, pruning) and avoid wetting leaves when watering.",
    "Keep tools clean and avoid working among wet plants to limit spread.",
    "Choose resistant varieties and keep the area free of fallen plant debris.",
    "Ask a local agricultural expert / extension office which treatment, if any, is suitable.",
]

# Each entry: keywords that must ALL appear in the normalised label.
_DB = [
    {
        "match": ["apple", "scab"],
        "title": "Apple Scab",
        "symptoms": "Olive-green to dark, scabby spots can appear on leaves and fruit. Severe leaf infection may cause yellowing and early leaf drop.",
        "cause": "Venturia inaequalis, a fungus that overwinters in infected fallen leaves.",
        "management": ["Remove fallen infected leaves and fruit.",
                       "Maintain an open canopy so leaves dry faster.",
                       "Use locally recommended resistant varieties and follow local extension guidance."],
    },
    {
        "match": ["apple", "black", "rot"],
        "title": "Apple Black Rot",
        "symptoms": "Leaves may develop small purple spots that enlarge into brown lesions; fruit can develop dark, circular rot.",
        "cause": "The fungus Diplodia seriata (also known as Botryosphaeria obtusa), which can persist in dead wood and mummified fruit.",
        "management": ["Remove mummified fruit and dead or infected branches.",
                       "Prune to improve airflow and reduce prolonged leaf wetness.",
                       "Ask a local extension service about controls appropriate for the orchard."],
    },
    {
        "match": ["cedar", "apple", "rust"],
        "title": "Cedar Apple Rust",
        "symptoms": "Bright yellow-orange spots may appear on the upper leaf surface, sometimes with tube-like structures on the underside.",
        "cause": "Gymnosporangium rust fungi that alternate between apple and juniper or cedar hosts.",
        "management": ["Remove nearby alternate hosts only where locally appropriate.",
                       "Choose resistant apple varieties when possible.",
                       "Seek local advice because the disease depends on both host plants and local conditions."],
    },
    {
        "match": ["corn", "gray", "leaf", "spot"],
        "title": "Corn Gray Leaf Spot",
        "symptoms": "Long, narrow, rectangular tan-to-gray lesions develop between the leaf veins, often on lower leaves first.",
        "cause": "Cercospora fungi that survive in infected corn residue.",
        "management": ["Rotate crops and manage infected residue where practical.",
                       "Use resistant hybrids and follow local agronomic guidance."],
    },
    {
        "match": ["corn", "common", "rust"],
        "title": "Corn Common Rust",
        "symptoms": "Small, raised reddish-brown to dark rust pustules can occur on both leaf surfaces.",
        "cause": "The fungus Puccinia sorghi.",
        "management": ["Use resistant hybrids where available.",
                       "Monitor crop development and consult local extension guidance for management decisions."],
    },
    {
        "match": ["northern", "leaf", "blight"],
        "title": "Northern Corn Leaf Blight",
        "symptoms": "Long, cigar-shaped gray-green lesions expand along the leaf and may merge when disease is severe.",
        "cause": "The fungus Exserohilum turcicum.",
        "management": ["Use resistant hybrids and rotate crops.",
                       "Manage infected residue and seek local agronomic advice."],
    },
    {
        "match": ["potato", "early", "blight"],
        "title": "Potato Early Blight",
        "symptoms": "Brown, often target-like spots with concentric rings can form on older leaves; surrounding tissue may yellow.",
        "cause": "Usually Alternaria solani, a fungus that can persist in crop debris.",
        "management": ["Rotate crops and remove infected plant debris.",
                       "Avoid prolonged leaf wetness and maintain balanced plant nutrition.",
                       "Consult local extension guidance for crop-specific control."],
    },
    {
        "match": ["potato", "late", "blight"],
        "title": "Potato Late Blight",
        "symptoms": "Water-soaked pale or dark lesions can enlarge rapidly; white growth may appear on leaf undersides in humid conditions.",
        "cause": "Phytophthora infestans, a water mould that can spread rapidly in cool, wet weather.",
        "management": ["Remove infected plants and volunteer potatoes according to local guidance.",
                       "Avoid handling plants when wet and monitor local disease alerts.",
                       "Contact an agricultural extension expert promptly if late blight is suspected."],
    },
    {
        "match": ["tomato", "early", "blight"],
        "title": "Tomato Early Blight",
        "symptoms": "Brown lesions with concentric rings often begin on older leaves; affected leaves may yellow and drop.",
        "cause": "Alternaria species, commonly Alternaria linariae (formerly A. tomatophila) or A. solani.",
        "management": ["Remove infected lower leaves and plant debris.",
                       "Rotate crops and water near the soil rather than over the foliage.",
                       "Follow local extension guidance for crop-specific control."],
    },
    {
        "match": ["tomato", "late", "blight"],
        "title": "Tomato Late Blight",
        "symptoms": "Irregular water-soaked green-to-brown lesions can enlarge quickly; humid conditions may produce pale growth on leaf undersides.",
        "cause": "Phytophthora infestans, a water mould that can spread quickly in cool, wet weather.",
        "management": ["Remove and dispose of infected plants according to local guidance.",
                       "Avoid overhead watering and monitor local disease alerts.",
                       "Seek prompt advice from an agricultural extension expert if late blight is suspected."],
    },
    {
        "match": ["tomato", "leaf", "mold"],
        "title": "Tomato Leaf Mold",
        "symptoms": "Pale green or yellow patches may appear on upper leaf surfaces, with olive-green to gray-purple fuzzy growth underneath in humid conditions.",
        "cause": "The fungus Passalora fulva (also called Fulvia fulva).",
        "management": ["Improve ventilation and reduce humidity around plants.",
                       "Avoid wetting foliage and remove badly affected plant debris.",
                       "Use resistant varieties where suitable."],
    },
    {
        "match": ["septoria", "leaf", "spot"],
        "title": "Septoria Leaf Spot",
        "symptoms": "Numerous small, round spots with pale centers and dark borders often appear on older leaves.",
        "cause": "The fungus Septoria lycopersici, spread by splashing water and infected debris.",
        "management": ["Remove lower infected leaves and plant debris.",
                       "Rotate crops and avoid overhead irrigation.",
                       "Clean tools and follow local extension advice."],
    },
    {
        "match": ["spider", "mites"],
        "title": "Two-Spotted Spider Mites",
        "symptoms": "Fine pale stippling, bronzing, and sometimes delicate webbing may appear, especially on leaf undersides.",
        "cause": "Feeding by the two-spotted spider mite, Tetranychus urticae.",
        "management": ["Inspect leaf undersides to confirm mites before acting.",
                       "Reduce plant stress and ask a local expert about appropriate integrated pest management."],
    },
    {
        "match": ["target", "spot"],
        "title": "Target Spot",
        "symptoms": "Brown circular lesions with concentric rings may develop on leaves, stems, or fruit.",
        "cause": "On tomato, target spot is commonly caused by the fungus Corynespora cassiicola.",
        "management": ["Remove infected debris and improve airflow.",
                       "Avoid prolonged leaf wetness and consult local crop guidance."],
    },
    {
        "match": ["yellow", "leaf", "curl"],
        "title": "Tomato Yellow Leaf Curl Disease",
        "symptoms": "Leaves may curl upward, become smaller or yellow, and plants may be stunted with reduced fruit set.",
        "cause": "Tomato yellow leaf curl viruses, commonly spread by whiteflies.",
        "management": ["Use clean planting material and manage weeds that can host whiteflies.",
                       "Use locally recommended whitefly-management practices and resistant varieties where available."],
    },
    {
        "match": ["tomato", "mosaic", "virus"],
        "title": "Tomato Mosaic Disease",
        "symptoms": "Leaves may show light and dark green mottling, distortion, or reduced growth.",
        "cause": "Tomato mosaic virus; it can spread through contaminated hands, tools, and plant material.",
        "management": ["Remove infected plants according to local guidance.",
                       "Wash hands and disinfect tools between plants.",
                       "Use clean seed and avoid handling plants after using tobacco products."],
    },
    {
        "match": ["bacterial", "spot"],
        "title": "Bacterial Spot",
        "symptoms": "Small dark, water-soaked spots may develop on leaves and fruit; leaf spots can become angular or develop yellow halos.",
        "cause": "Bacterial species in the Xanthomonas group; exact species depends on the crop.",
        "management": ["Use clean planting material and avoid handling wet plants.",
                       "Remove infected debris and avoid overhead irrigation.",
                       "Consult an expert for crop-specific control options."],
    },
    {
        "match": ["black", "rot"],
        "title": "Black Rot",
        "symptoms": "Small brown circular spots on leaves that enlarge, with a dark border and tiny black dots (fruiting bodies) in the centre; fruit may shrivel into hard black 'mummies'.",
        "cause": "Fungus Guignardia bidwellii (grape black rot); favoured by warm, wet weather.",
        "management": ["Remove mummified fruit and infected leaves/canes.",
                       "Prune for open canopy so foliage dries quickly.",
                       "Follow local expert guidance on protective spray programmes."],
    },
    {
        "match": ["esca"],
        "title": "Esca (Black Measles)",
        "symptoms": "Yellow or brown 'tiger-stripe' patterns between leaf veins, dark spots on berries, and wood decay in older vines.",
        "cause": "A complex of wood-inhabiting fungi that enter through pruning wounds.",
        "management": ["Prune in dry weather and protect large pruning wounds.",
                       "Remove and destroy dead or badly affected vines/wood.",
                       "Consult an expert - there is no simple leaf-only cure."],
    },
    {
        "match": ["black", "measles"],
        "title": "Esca (Black Measles)",
        "symptoms": "Yellow or brown 'tiger-stripe' patterns between leaf veins, dark spots on berries, and wood decay in older vines.",
        "cause": "A complex of wood-inhabiting fungi that enter through pruning wounds.",
        "management": ["Prune in dry weather and protect large pruning wounds.",
                       "Remove and destroy dead or badly affected vines/wood."],
    },
    {
        "match": ["grape", "blight"],
        "title": "Grape Leaf Blight",
        "symptoms": "Irregular reddish-brown to dark patches on leaves that may merge and cause early leaf drop.",
        "cause": "Fungus Pseudocercospora vitis (Isariopsis leaf spot) in humid conditions.",
        "management": ["Remove infected leaves and fallen debris.",
                       "Improve canopy airflow and avoid overhead irrigation."],
    },
    {
        "match": ["powdery", "mildew"],
        "title": "Powdery Mildew",
        "symptoms": "White to grey powdery patches on the leaf surface, which may curl, yellow or dry out.",
        "cause": "Powdery mildew fungi (order Erysiphales); thrives in warm days, cool humid nights and crowded foliage.",
        "management": ["Prune for airflow and avoid excess nitrogen fertiliser.",
                       "Remove badly affected leaves.",
                       "Ask an expert about suitable preventive treatment for the crop."],
    },
    {
        "match": ["bacterial", "spot"],
        "title": "Bacterial Spot",
        "symptoms": "Small dark, water-soaked angular spots that may turn purple-brown and drop out, leaving a 'shot-hole' look.",
        "cause": "Bacteria (e.g. Xanthomonas species); spread by rain splash and wet foliage.",
        "management": ["Avoid overhead watering and working in wet foliage.",
                       "Remove infected leaves; use clean planting material.",
                       "Consult an expert for crop-specific control options."],
    },
    {
        "match": ["leaf", "spot"],
        "title": "Leaf Spot",
        "symptoms": "Round to irregular brown or dark spots on leaves, sometimes with a yellow halo.",
        "cause": "Usually fungal or bacterial pathogens; exact cause depends on the plant and dataset class.",
        "management": GENERAL_MANAGEMENT[:4],
    },
    {
        "match": ["rust"],
        "title": "Rust",
        "symptoms": "Orange, yellow or brown pustules (often on the underside of leaves) and yellowing of the leaf above.",
        "cause": "Rust fungi (order Pucciniales).",
        "management": GENERAL_MANAGEMENT[:4],
    },
    {
        "match": ["scab"],
        "title": "Scab",
        "symptoms": "Olive-green to dark, velvety or scabby lesions on leaves and fruit.",
        "cause": "Fungal infection favoured by cool, wet spring weather.",
        "management": GENERAL_MANAGEMENT[:4],
    },
]


def pretty_label(label: str) -> str:
    """'Grape___Black_rot' -> 'Grape - Black rot'; 'Grape_Black_Rot' -> 'Grape Black Rot'."""
    s = label.strip()
    s = re.sub(r"_{2,}", " - ", s)
    s = s.replace("_", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return " ".join(w[:1].upper() + w[1:] if w.islower() else w for w in s.split(" "))


def get_disease_info(label: str) -> dict:
    """Return display info for a predicted class label."""
    norm = re.sub(r"[^a-z]+", " ", label.lower())
    name = pretty_label(label)

    if "healthy" in norm:
        return {
            "name": "Healthy Leaf", "known": True,
            "symptoms": "No disease symptoms were recognised by the classifier.",
            "cause": "Not applicable.",
            "management": ["Keep monitoring the plant regularly.",
                           "Maintain good watering, spacing and sanitation to prevent disease."],
            "disclaimer": DISCLAIMER,
        }

    for entry in _DB:
        if all(k in norm.split() or k in norm for k in entry["match"]):
            return {"name": name, "known": True, "symptoms": entry["symptoms"],
                    "cause": entry["cause"], "management": entry["management"],
                    "disclaimer": DISCLAIMER}

    return {
        "name": name, "known": False,
        "symptoms": "The classifier selected this class, but crop-specific symptom information is not available in the project.",
        "cause": "Not available for this class in the project database.",
        "management": GENERAL_MANAGEMENT,
        "disclaimer": DISCLAIMER,
    }
