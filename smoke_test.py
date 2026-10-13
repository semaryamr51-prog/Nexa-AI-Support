"""Fast offline end-to-end checks, independent from optional cloud services."""
from pathlib import Path
from src.agents import route_and_run
from src.classify import classify
from src.knowledge import load_passages
from src.rag import get_retriever


def main():
    print('Nexa smoke tests')
    assert Path('models/classifier.pt').exists(), 'Run python -m src.train first.'
    assert len(load_passages()) >= 10, 'Knowledge docs are missing'
    assert get_retriever().invoke('refund')
    for request, expected in [
        ('Why was invoice INV-2091 charged?', 'billing'),
        ('The video player is not working', 'technical'),
        ('Can I book a session next Monday?', 'booking'),
        ('Do I get a course certificate?', 'general'),
    ]:
        intent, confidence = classify(request)
        response = route_and_run(request)
        assert response['reply'] and response['agent'] in {'billing', 'technical', 'booking', 'general'}
        assert 0 <= confidence <= 1
        print(f'  {expected:10s} -> {intent:10s} ({confidence:.0%}) agent={response["agent"]}; {response["provider"]}')
    print('All offline smoke tests passed.')

if __name__ == '__main__':
    main()
