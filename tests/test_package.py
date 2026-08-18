def test_package_can_be_imported() -> None:
    import company_researcher

    assert company_researcher.__doc__
