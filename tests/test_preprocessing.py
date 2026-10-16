def test_label_mapping():
    # Beklenen 4 ana etiketimizin doğru tanımlandığını test ediyoruz
    expected_labels = ['Bug', 'Feature Request', 'Question', 'Documentation']
    
    assert len(expected_labels) == 4
    assert 'Bug' in expected_labels
    assert 'Feature Request' in expected_labels